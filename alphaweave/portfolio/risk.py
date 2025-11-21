"""Risk estimation helpers for portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: str = "sample",
    span: int = 60,
) -> pd.DataFrame:
    """
    Estimate asset return covariance matrix.

    Args:
        returns: DataFrame with columns = assets, index = dates
        method: Estimation method ("sample" or "ewma")
        span: Span parameter for EWMA (default: 60)

    Returns:
        Covariance matrix DataFrame (index and columns = assets)
    """
    if method == "sample":
        return returns.cov()
    elif method == "ewma":
        # Exponentially weighted moving average covariance
        # Use pandas ewm with halflife
        halflife = span / np.log(2)
        # Center returns around EWMA mean
        ewma_mean = returns.ewm(halflife=halflife, adjust=False).mean()
        centered = returns - ewma_mean
        
        # Compute EWMA covariance using manual calculation
        # For each pair of assets, compute EWMA of product
        n_assets = len(returns.columns)
        cov_matrix = np.zeros((n_assets, n_assets))
        assets = list(returns.columns)
        
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i <= j:
                    # Compute EWMA of centered_i * centered_j
                    product = centered[asset_i] * centered[asset_j]
                    ewma_cov = product.ewm(halflife=halflife, adjust=False).mean().iloc[-1]
                    cov_matrix[i, j] = ewma_cov
                    if i != j:
                        cov_matrix[j, i] = ewma_cov  # Symmetric
        
        final_cov = pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns,
        )
        return final_cov
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sample' or 'ewma'")


def estimate_volatility(
    returns: pd.DataFrame,
    *,
    method: str = "sample",
    span: int = 60,
    trading_days: int = 252,
) -> pd.Series:
    """
    Per-asset volatility estimate.

    Args:
        returns: DataFrame with columns = assets, index = dates
        method: Estimation method ("sample" or "ewma")
        span: Span parameter for EWMA (default: 60)
        trading_days: Trading days per year for annualization (default: 252)

    Returns:
        Series of annualized volatilities (index = assets)
    """
    if method == "sample":
        vol = returns.std() * np.sqrt(trading_days)
    elif method == "ewma":
        halflife = span / np.log(2)
        vol = returns.ewm(halflife=halflife, adjust=False).std().iloc[-1] * np.sqrt(trading_days)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sample' or 'ewma'")

    return vol

