"""Barra-lite multi-factor risk model."""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class RiskModel:
    """
    Multi-factor risk model.

    Attributes:
        exposures: Factor exposures matrix [symbols × factors]
        factor_cov: Factor covariance matrix [factors × factors]
        specific_var: Specific (idiosyncratic) variance per symbol [symbols]
    """

    exposures: pd.DataFrame  # index=symbol, columns=factors
    factor_cov: pd.DataFrame  # factors × factors
    specific_var: pd.Series  # per symbol

    def total_covariance(self) -> pd.DataFrame:
        """
        Compute total asset covariance matrix.

        Formula: B * F * B' + D
        Where:
            B = exposures matrix
            F = factor covariance
            D = diagonal matrix of specific variances

        Returns:
            Total covariance matrix [symbols × symbols]
        """
        B = self.exposures.values
        F = self.factor_cov.values
        D = np.diag(self.specific_var.values)

        # B * F * B' + D
        total_cov = B @ F @ B.T + D

        return pd.DataFrame(
            total_cov,
            index=self.exposures.index,
            columns=self.exposures.index,
        )

    def validate(self) -> bool:
        """
        Validate risk model consistency.

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check dimensions
        if self.exposures.shape[1] != self.factor_cov.shape[0]:
            raise ValueError(
                f"Exposures columns ({self.exposures.shape[1]}) must match "
                f"factor_cov rows ({self.factor_cov.shape[0]})"
            )

        if self.exposures.shape[1] != self.factor_cov.shape[1]:
            raise ValueError(
                f"Factor covariance must be square, got {self.factor_cov.shape}"
            )

        if len(self.specific_var) != len(self.exposures):
            raise ValueError(
                f"Specific variance length ({len(self.specific_var)}) must match "
                f"exposures rows ({len(self.exposures)})"
            )

        # Check alignment
        if not self.exposures.index.equals(self.specific_var.index):
            raise ValueError("Exposures and specific_var must have same index")

        # Check factor names
        if not self.exposures.columns.equals(self.factor_cov.index):
            raise ValueError("Exposures columns must match factor_cov index")

        if not self.factor_cov.index.equals(self.factor_cov.columns):
            raise ValueError("Factor covariance must have matching index and columns")

        return True


@dataclass
class RiskDecomposition:
    """
    Risk decomposition for a portfolio.

    Attributes:
        total_vol: Total portfolio volatility (annualized)
        factor_contrib: Contribution of each factor to portfolio variance
        specific_contrib: Specific variance contribution per asset
        marginal_risk: Marginal risk contribution per asset
        component_risk: Component risk (w_i * marginal_risk_i) per asset
    """

    total_vol: float
    factor_contrib: pd.Series  # factor → contribution
    specific_contrib: pd.Series  # symbol → contribution
    marginal_risk: pd.Series  # symbol → marginal risk
    component_risk: pd.Series  # symbol → component risk

    def __repr__(self) -> str:
        return (
            f"RiskDecomposition(total_vol={self.total_vol:.4f}, "
            f"factor_risk={self.factor_contrib.sum():.4f}, "
            f"specific_risk={self.specific_contrib.sum():.4f})"
        )


def estimate_factor_returns(
    returns: pd.DataFrame,
    exposures: Dict[str, pd.DataFrame],
    *,
    min_assets: int = 20,
) -> pd.DataFrame:
    """
    Estimate factor returns using cross-sectional regression (Fama-MacBeth style).

    For each date t:
        R_it = α_t + β_i1 f1_t + β_i2 f2_t + ... + ε_it

    We solve: factor_returns_t = (X_t' X_t)^(-1) X_t' R_t

    Args:
        returns: Asset returns [date × symbol]
        exposures: Dictionary of factor name → exposure DataFrame [date × symbol]
        min_assets: Minimum number of assets required for regression (default: 20)

    Returns:
        Factor returns DataFrame [date × factor]
    """
    if not exposures:
        return pd.DataFrame()

    # Align all exposure DataFrames
    factor_names = list(exposures.keys())
    all_dates = set(returns.index)
    for exp_df in exposures.values():
        all_dates = all_dates.intersection(set(exp_df.index))

    dates = sorted(all_dates)

    factor_returns = {name: [] for name in factor_names}
    valid_dates = []

    for date in dates:
        # Get returns for this date
        r_t = returns.loc[date].dropna()
        if len(r_t) < min_assets:
            continue

        # Build exposure matrix X_t
        X_cols = {}
        for name in factor_names:
            X_cols[name] = exposures[name].loc[date].dropna()

        # Align X and y
        common_symbols = r_t.index
        for name in factor_names:
            common_symbols = common_symbols.intersection(X_cols[name].index)

        if len(common_symbols) < min_assets:
            continue

        # Extract aligned data
        y = r_t.loc[common_symbols].values
        X_matrix = np.column_stack(
            [X_cols[name].loc[common_symbols].values for name in factor_names]
        )

        # Add intercept
        X_matrix = np.column_stack([np.ones(len(X_matrix)), X_matrix])

        # OLS: beta = (X'X)^(-1)X'y
        try:
            beta = np.linalg.lstsq(X_matrix, y, rcond=None)[0]
            # Factor returns are coefficients (skip intercept)
            for idx, name in enumerate(factor_names):
                factor_returns[name].append(beta[idx + 1])
            valid_dates.append(date)
        except np.linalg.LinAlgError:
            continue

    if not valid_dates:
        return pd.DataFrame()

    return pd.DataFrame(
        {name: factor_returns[name] for name in factor_names},
        index=valid_dates,
    )


def estimate_factor_covariance(
    factor_returns: pd.DataFrame,
    method: str = "shrinkage_lw",
    span: int = 60,
) -> pd.DataFrame:
    """
    Estimate factor covariance matrix.

    Args:
        factor_returns: Factor returns [date × factor]
        method: Estimation method:
            - "sample": Sample covariance
            - "ewma": Exponentially weighted moving average
            - "shrinkage_lw": Ledoit-Wolf shrinkage
            - "shrinkage_oas": Oracle Approximating Shrinkage
        span: Span parameter for EWMA (default: 60)

    Returns:
        Factor covariance matrix [factors × factors]
    """
    if factor_returns.empty:
        return pd.DataFrame()

    if method == "sample":
        cov = factor_returns.cov()
    elif method == "ewma":
        from alphaweave.portfolio.risk import estimate_covariance

        cov = estimate_covariance(factor_returns, method="ewma", span=span)
    elif method == "shrinkage_lw":
        try:
            from alphaweave.pipeline.covariance import shrink_cov_lw
        except ImportError:
            raise ImportError("shrink_cov_lw not available. Install Sprint 13 components.")
        sample_cov = factor_returns.cov()
        cov = shrink_cov_lw(sample_cov)
    elif method == "shrinkage_oas":
        try:
            from alphaweave.pipeline.covariance import shrink_cov_oas
        except ImportError:
            raise ImportError("shrink_cov_oas not available. Install Sprint 13 components.")
        sample_cov = factor_returns.cov()
        cov = shrink_cov_oas(sample_cov)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use 'sample', 'ewma', 'shrinkage_lw', or 'shrinkage_oas'"
        )

    return cov


def compute_exposures(
    factor_data: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute factor exposures for a specific date.

    Args:
        factor_data: Dictionary of factor name → factor DataFrame [date × symbol]
        date: Date to extract exposures

    Returns:
        Exposures matrix [symbol × factor]
    """
    exposures = {}
    common_symbols = None

    for factor_name, factor_df in factor_data.items():
        if date not in factor_df.index:
            continue

        factor_values = factor_df.loc[date].dropna()
        if common_symbols is None:
            common_symbols = set(factor_values.index)
        else:
            common_symbols = common_symbols.intersection(set(factor_values.index))

        exposures[factor_name] = factor_values

    if not exposures or common_symbols is None:
        return pd.DataFrame()

    # Align all factors to common symbols
    aligned_exposures = {}
    for factor_name, factor_values in exposures.items():
        aligned_exposures[factor_name] = factor_values.loc[list(common_symbols)]

    return pd.DataFrame(aligned_exposures, index=list(common_symbols))


def compute_exposures_rolling(
    factor_data: Dict[str, pd.DataFrame],
    window: int = 20,
    date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute rolling average exposures.

    Args:
        factor_data: Dictionary of factor name → factor DataFrame [date × symbol]
        window: Rolling window size (default: 20)
        date: End date (default: last date in data)

    Returns:
        Exposures matrix [symbol × factor]
    """
    if date is None:
        # Use last date available
        dates = set()
        for factor_df in factor_data.values():
            dates.update(factor_df.index)
        date = max(dates)

    # Get rolling window
    exposures_list = []
    for factor_name, factor_df in factor_data.items():
        if date not in factor_df.index:
            continue

        # Get window of data
        window_data = factor_df.loc[:date].iloc[-window:]
        # Average over window
        avg_exposure = window_data.mean()
        exposures_list.append((factor_name, avg_exposure))

    if not exposures_list:
        return pd.DataFrame()

    # Align to common symbols
    common_symbols = set(exposures_list[0][1].index)
    for _, exp_series in exposures_list[1:]:
        common_symbols = common_symbols.intersection(set(exp_series.index))

    aligned = {}
    for factor_name, exp_series in exposures_list:
        aligned[factor_name] = exp_series.loc[list(common_symbols)]

    return pd.DataFrame(aligned, index=list(common_symbols))


def estimate_specific_risk(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    window: int = 252,
) -> pd.Series:
    """
    Estimate specific (idiosyncratic) risk from regression residuals.

    For each asset i:
        ε_it = R_it - Σ_k β_ik * f_k,t
        specific_var_i = Var(ε_i)

    Args:
        returns: Asset returns [date × symbol]
        factor_returns: Factor returns [date × factor]
        exposures: Factor exposures [symbol × factor]
        window: Rolling window for variance estimation (default: 252)

    Returns:
        Specific variance per symbol [symbol]
    """
    # Align dates
    common_dates = set(returns.index).intersection(set(factor_returns.index))
    if not common_dates:
        return pd.Series(dtype=float)

    dates = sorted(common_dates)
    symbols = exposures.index

    specific_vars = {}

    for symbol in symbols:
        if symbol not in returns.columns:
            continue

        # Get asset returns
        asset_returns = returns.loc[dates, symbol].dropna()
        if len(asset_returns) < window:
            continue

        # Get exposures for this asset
        asset_exposures = exposures.loc[symbol]

        # Compute predicted returns: Σ_k β_ik * f_k,t
        predicted = pd.Series(0.0, index=asset_returns.index)
        for factor_name, exposure in asset_exposures.items():
            if factor_name in factor_returns.columns:
                factor_ret = factor_returns.loc[asset_returns.index, factor_name]
                predicted += exposure * factor_ret

        # Compute residuals
        residuals = asset_returns - predicted

        # Estimate variance of residuals (rolling)
        if len(residuals) >= window:
            specific_var = residuals.rolling(window=window).var().iloc[-1]
        else:
            specific_var = residuals.var()

        # Annualize (assuming daily returns)
        specific_vars[symbol] = specific_var * 252 if not np.isnan(specific_var) else 0.0

    return pd.Series(specific_vars)


def decompose_risk(
    weights: pd.Series,
    model: RiskModel,
) -> RiskDecomposition:
    """
    Decompose portfolio risk into factor and specific contributions.

    Formula:
        Total Cov = B * F * B' + D
        Total Vol = sqrt(w' * Cov * w)

    Factor contribution:
        factor_contrib = (w' * B) * F * (B' * w)

    Specific contribution:
        specific_contrib = w^2 * specific_var

    Args:
        weights: Portfolio weights [symbol]
        model: Risk model

    Returns:
        Risk decomposition
    """
    # Validate alignment
    if not weights.index.equals(model.exposures.index):
        # Align weights to exposures
        weights = weights.reindex(model.exposures.index).fillna(0.0)

    # Convert to numpy arrays
    w = weights.values
    B = model.exposures.values
    F = model.factor_cov.values
    D = np.diag(model.specific_var.values)

    # Total covariance
    total_cov = B @ F @ B.T + D

    # Total variance
    total_var = w.T @ total_cov @ w
    total_vol = np.sqrt(total_var)

    # Factor contribution
    # w' * B gives factor exposures of portfolio
    portfolio_factor_exposures = w.T @ B  # [1 × factors]

    # Factor variance contribution
    factor_var_contrib = portfolio_factor_exposures @ F @ portfolio_factor_exposures.T
    factor_var_contrib = factor_var_contrib[0, 0]

    # Per-factor contribution
    factor_contrib = {}
    for idx, factor_name in enumerate(model.factor_cov.index):
        # Contribution of this factor
        factor_exposure = portfolio_factor_exposures[0, idx]
        factor_var = F[idx, idx]
        factor_contrib[factor_name] = factor_exposure**2 * factor_var

    factor_contrib_series = pd.Series(factor_contrib)

    # Specific contribution
    specific_contrib = w**2 * model.specific_var.values
    specific_contrib_series = pd.Series(specific_contrib, index=weights.index)

    # Marginal risk: d(vol)/d(w_i) = (Cov * w)_i / vol
    marginal_risk = (total_cov @ w) / total_vol
    marginal_risk_series = pd.Series(marginal_risk, index=weights.index)

    # Component risk: w_i * marginal_risk_i
    component_risk = w * marginal_risk
    component_risk_series = pd.Series(component_risk, index=weights.index)

    return RiskDecomposition(
        total_vol=total_vol,
        factor_contrib=factor_contrib_series,
        specific_contrib=specific_contrib_series,
        marginal_risk=marginal_risk_series,
        component_risk=component_risk_series,
    )


def hedge_exposures(
    weights: pd.Series,
    exposures: pd.DataFrame,
    neutralize_factors: list[str],
) -> pd.Series:
    """
    Neutralize portfolio exposure to specified factors.

    Args:
        weights: Current portfolio weights [symbol]
        exposures: Factor exposures [symbol × factor]
        neutralize_factors: List of factor names to neutralize

    Returns:
        Adjusted weights with neutralized exposures
    """
    # Align weights to exposures
    if not weights.index.equals(exposures.index):
        weights = weights.reindex(exposures.index).fillna(0.0)

    # Get current factor exposures
    portfolio_exposures = weights @ exposures

    # For each factor to neutralize, adjust weights
    adjusted_weights = weights.copy()

    for factor_name in neutralize_factors:
        if factor_name not in exposures.columns:
            continue

        # Current exposure to this factor
        current_exposure = portfolio_exposures[factor_name]

        # Factor loadings
        factor_loadings = exposures[factor_name]

        # Compute adjustment to neutralize
        # We want: (w + delta) @ factor_loadings = 0
        # So: w @ factor_loadings + delta @ factor_loadings = 0
        # delta @ factor_loadings = -current_exposure
        # If we use delta = -current_exposure * factor_loadings / ||factor_loadings||^2
        # Then: delta @ factor_loadings = -current_exposure * (factor_loadings @ factor_loadings) / ||factor_loadings||^2
        #      = -current_exposure

        factor_norm_sq = (factor_loadings**2).sum()
        if factor_norm_sq > 1e-10:
            adjustment = -current_exposure * factor_loadings / factor_norm_sq
            adjusted_weights = adjusted_weights + adjustment

    # Renormalize to sum to 1 (preserve long-only if needed)
    if adjusted_weights.sum() != 0:
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

    return adjusted_weights

