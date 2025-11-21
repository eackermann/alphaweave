"""Rolling regressions and factor return computation."""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.pipeline.factors import Factor


class RollingOLSRegressor:
    """Rolling OLS regression for factor models."""

    def __init__(
        self,
        dependent: Factor,
        independents: Dict[str, Factor],
        window: int = 252,
    ):
        """
        Initialize rolling OLS regressor.

        Args:
            dependent: Dependent factor (y)
            independents: Dictionary of independent factor names to Factor instances (X)
            window: Rolling window size (default: 252)
        """
        self.dependent = dependent
        self.independents = independents
        self.window = window

    def fit(self, data: Dict[str, Frame]) -> Dict[str, pd.DataFrame]:
        """
        Fit rolling regressions.

        Returns:
            Dictionary with keys:
            - "coefficients": DataFrame of factor loadings [datetime × factors]
            - "residuals": DataFrame of residuals [datetime × symbols]
            - "predicted": DataFrame of predicted values [datetime × symbols]
            - "r_squared": DataFrame of R² values [datetime × symbols]
        """
        # Compute dependent variable
        y_df = self.dependent.compute(data)

        # Compute independent variables
        x_dfs = {}
        for name, factor in self.independents.items():
            x_dfs[name] = factor.compute(data)

        # Align all DataFrames
        all_dfs = {"y": y_df, **x_dfs}
        aligned = self._align_dataframes(all_dfs)

        if aligned["y"].empty:
            return {
                "coefficients": pd.DataFrame(),
                "residuals": pd.DataFrame(),
                "predicted": pd.DataFrame(),
                "r_squared": pd.DataFrame(),
            }

        # Get symbols
        symbols = aligned["y"].columns

        # Initialize results
        coefficients = {name: [] for name in x_dfs.keys()}
        residuals_dict = {symbol: [] for symbol in symbols}
        predicted_dict = {symbol: [] for symbol in symbols}
        r_squared_dict = {symbol: [] for symbol in symbols}
        dates = []

        # Rolling regression
        for i in range(len(aligned["y"])):
            if i < self.window - 1:
                # Not enough data yet
                continue

            window_y = aligned["y"].iloc[i - self.window + 1 : i + 1]
            window_x = {name: aligned[name].iloc[i - self.window + 1 : i + 1] for name in x_dfs.keys()}

            date = aligned["y"].index[i]
            dates.append(date)

            # Fit regression for each symbol
            for symbol in symbols:
                y_vals = window_y[symbol].dropna()
                if len(y_vals) < 2:
                    for name in x_dfs.keys():
                        coefficients[name].append(np.nan)
                    residuals_dict[symbol].append(np.nan)
                    predicted_dict[symbol].append(np.nan)
                    r_squared_dict[symbol].append(np.nan)
                    continue

                # Build X matrix
                x_vals = {}
                for name in x_dfs.keys():
                    x_vals[name] = window_x[name][symbol].dropna()

                # Align X and y
                common_idx = y_vals.index
                for name in x_dfs.keys():
                    common_idx = common_idx.intersection(x_vals[name].index)

                if len(common_idx) < 2:
                    for name in x_dfs.keys():
                        coefficients[name].append(np.nan)
                    residuals_dict[symbol].append(np.nan)
                    predicted_dict[symbol].append(np.nan)
                    r_squared_dict[symbol].append(np.nan)
                    continue

                y_aligned = y_vals.loc[common_idx].values
                X_matrix = np.column_stack([x_vals[name].loc[common_idx].values for name in x_dfs.keys()])

                # Add intercept
                X_matrix = np.column_stack([np.ones(len(X_matrix)), X_matrix])

                # OLS: beta = (X'X)^(-1)X'y
                try:
                    beta = np.linalg.lstsq(X_matrix, y_aligned, rcond=None)[0]
                    y_pred = X_matrix @ beta
                    residuals = y_aligned - y_pred

                    # R²
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_aligned - np.mean(y_aligned)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    # Store coefficients (skip intercept)
                    for idx, name in enumerate(x_dfs.keys()):
                        coefficients[name].append(beta[idx + 1])

                    residuals_dict[symbol].append(residuals[-1])
                    predicted_dict[symbol].append(y_pred[-1])
                    r_squared_dict[symbol].append(r2)

                except np.linalg.LinAlgError:
                    for name in x_dfs.keys():
                        coefficients[name].append(np.nan)
                    residuals_dict[symbol].append(np.nan)
                    predicted_dict[symbol].append(np.nan)
                    r_squared_dict[symbol].append(np.nan)

        # Convert to DataFrames
        coeff_df = pd.DataFrame(coefficients, index=dates)
        residuals_df = pd.DataFrame(residuals_dict, index=dates)
        predicted_df = pd.DataFrame(predicted_dict, index=dates)
        r_squared_df = pd.DataFrame(r_squared_dict, index=dates)

        return {
            "coefficients": coeff_df,
            "residuals": residuals_df,
            "predicted": predicted_df,
            "r_squared": r_squared_df,
        }

    def _align_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align multiple DataFrames to common index."""
        if not dfs:
            return {}

        # Get all indices
        all_indices = set()
        for df in dfs.values():
            all_indices.update(df.index)

        master_index = pd.DatetimeIndex(sorted(all_indices))

        # Align each DataFrame
        aligned = {}
        for name, df in dfs.items():
            aligned[name] = df.reindex(master_index)

        return aligned


def compute_factor_returns(
    factors: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute factor returns using cross-sectional regression (Fama-MacBeth style).

    Args:
        factors: Dictionary of factor name -> DataFrame [datetime × symbols]
        returns: Returns DataFrame [datetime × symbols]

    Returns:
        DataFrame of factor returns [datetime × factors]
    """
    # Align all DataFrames
    all_dfs = {"returns": returns, **factors}
    aligned = _align_dataframes(all_dfs)

    if aligned["returns"].empty:
        return pd.DataFrame()

    dates = aligned["returns"].index
    factor_names = list(factors.keys())

    factor_returns = {name: [] for name in factor_names}
    factor_returns["dates"] = []

    # Cross-sectional regression for each date
    for date in dates:
        y = aligned["returns"].loc[date].dropna()
        if len(y) < 2:
            continue

        # Build X matrix
        X_cols = {}
        for name in factor_names:
            X_cols[name] = aligned[name].loc[date].dropna()

        # Align X and y
        common_idx = y.index
        for name in factor_names:
            common_idx = common_idx.intersection(X_cols[name].index)

        if len(common_idx) < len(factor_names) + 1:
            continue

        y_aligned = y.loc[common_idx].values
        X_matrix = np.column_stack([X_cols[name].loc[common_idx].values for name in factor_names])

        # Add intercept
        X_matrix = np.column_stack([np.ones(len(X_matrix)), X_matrix])

        # OLS
        try:
            beta = np.linalg.lstsq(X_matrix, y_aligned, rcond=None)[0]
            # Factor returns are coefficients (skip intercept)
            for idx, name in enumerate(factor_names):
                factor_returns[name].append(beta[idx + 1])
            factor_returns["dates"].append(date)
        except np.linalg.LinAlgError:
            continue

    # Convert to DataFrame
    if not factor_returns["dates"]:
        return pd.DataFrame()

    result = pd.DataFrame(
        {name: factor_returns[name] for name in factor_names},
        index=factor_returns["dates"],
    )
    return result


def _align_dataframes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align multiple DataFrames to common index."""
    if not dfs:
        return {}

    all_indices = set()
    for df in dfs.values():
        all_indices.update(df.index)

    master_index = pd.DatetimeIndex(sorted(all_indices))

    aligned = {}
    for name, df in dfs.items():
        aligned[name] = df.reindex(master_index)

    return aligned

