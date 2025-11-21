"""Factor regression and decomposition for alphaweave."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorRegressionResult:
    """Result of factor regression analysis."""

    alpha: float
    betas: pd.Series  # index = factor names
    residual_std: float
    r2: float
    tstats: pd.Series  # index = ["alpha"] + factor names
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"FactorRegressionResult(alpha={self.alpha:.4f}, "
            f"r2={self.r2:.4f}, n_obs={self.n_obs})"
        )


def factor_regression(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    add_constant: bool = True,
) -> FactorRegressionResult:
    """
    Run a simple OLS regression of strategy returns on factor returns.

    Args:
        strategy_returns: Strategy returns series (indexed by datetime)
        factor_returns: Factor returns DataFrame (columns = factor names, index = datetime)
        add_constant: If True, add intercept term (alpha)

    Returns:
        FactorRegressionResult with alpha, betas, R², t-stats, etc.
    """
    # Align on intersection of dates
    common_dates = strategy_returns.index.intersection(factor_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between strategy and factor returns")

    y = strategy_returns.loc[common_dates].values
    X = factor_returns.loc[common_dates].values

    # Add constant term if requested
    if add_constant:
        X = np.column_stack([np.ones(len(X)), X])
        factor_names = ["alpha"] + list(factor_returns.columns)
    else:
        factor_names = list(factor_returns.columns)

    # OLS regression using numpy
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback if singular matrix
        coeffs = np.zeros(X.shape[1])
        residuals = y
        rank = 0
        s = None

    # Extract alpha and betas
    if add_constant:
        alpha = float(coeffs[0])
        betas_values = coeffs[1:]
    else:
        alpha = 0.0
        betas_values = coeffs

    betas = pd.Series(betas_values, index=factor_returns.columns)

    # Compute R²
    y_mean = np.mean(y)
    ss_res = np.sum(residuals**2) if len(residuals) > 0 else np.sum((y - np.dot(X, coeffs)) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute residual standard deviation
    n_obs = len(y)
    n_params = len(coeffs)
    residual_std = np.sqrt(ss_res / (n_obs - n_params)) if (n_obs - n_params) > 0 else 0.0

    # Compute t-statistics
    if residual_std > 0 and s is not None and len(s) > 0:
        # Standard errors from covariance matrix
        try:
            cov = np.linalg.inv(X.T @ X) * (residual_std**2)
            se = np.sqrt(np.diag(cov))
            tstats = coeffs / se
        except np.linalg.LinAlgError:
            # Fallback: simple t-stats
            se = np.ones(len(coeffs)) * residual_std
            tstats = coeffs / se
    else:
        tstats = np.zeros(len(coeffs))

    tstats_series = pd.Series(tstats, index=factor_names)

    return FactorRegressionResult(
        alpha=alpha,
        betas=betas,
        residual_std=float(residual_std),
        r2=float(r2),
        tstats=tstats_series,
        n_obs=n_obs,
    )

