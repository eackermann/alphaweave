"""Statistical transforms and preprocessing for factors."""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from alphaweave.pipeline.expressions import FactorExpression


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorize a DataFrame by capping values at percentiles.

    Args:
        df: Input DataFrame
        lower: Lower percentile threshold (default: 0.01)
        upper: Upper percentile threshold (default: 0.99)

    Returns:
        Winsorized DataFrame
    """
    result = df.copy()
    for col in df.columns:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        result[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return result


def normalize(df: pd.DataFrame, method: Literal["minmax", "zscore"] = "minmax") -> pd.DataFrame:
    """
    Normalize a DataFrame.

    Args:
        df: Input DataFrame
        method: Normalization method - "minmax" or "zscore" (default: "minmax")

    Returns:
        Normalized DataFrame
    """
    result = df.copy()
    if method == "minmax":
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                result[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                result[col] = 0.0
    elif method == "zscore":
        for col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std != 0:
                result[col] = (df[col] - col_mean) / col_std
            else:
                result[col] = 0.0
    return result


def lag(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Lag a DataFrame by specified periods.

    Args:
        df: Input DataFrame
        periods: Number of periods to lag (default: 1)

    Returns:
        Lagged DataFrame
    """
    return df.shift(periods=periods)


def smooth(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Smooth a DataFrame using moving average.

    Args:
        df: Input DataFrame
        window: Moving average window (default: 3)

    Returns:
        Smoothed DataFrame
    """
    return df.rolling(window=window, min_periods=1).mean()


# Extend FactorExpression with new transforms
def _add_transforms_to_expression():
    """Add new transform methods to FactorExpression."""

    def winsorize_method(self, lower: float = 0.01, upper: float = 0.99) -> "FactorExpression":
        """Apply winsorization to factor."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: winsorize(df, lower=lower, upper=upper))
        return expr

    def normalize_method(self, method: Literal["minmax", "zscore"] = "minmax") -> "FactorExpression":
        """Apply normalization to factor."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: normalize(df, method=method))
        return expr

    def lag_method(self, periods: int = 1) -> "FactorExpression":
        """Apply lag to factor."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: lag(df, periods=periods))
        return expr

    def smooth_method(self, window: int = 3) -> "FactorExpression":
        """Apply smoothing to factor."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: smooth(df, window=window))
        return expr

    def rolling_zscore_method(self, window: int) -> "FactorExpression":
        """Apply rolling z-score normalization."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: df.rolling(window=window, min_periods=1).apply(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
        ))
        return expr

    # Add methods to FactorExpression class
    FactorExpression.winsorize = winsorize_method
    FactorExpression.normalize = normalize_method
    FactorExpression.lag = lag_method
    FactorExpression.smooth = smooth_method
    FactorExpression.rolling_zscore = rolling_zscore_method


# Initialize transforms when module is imported
_add_transforms_to_expression()

