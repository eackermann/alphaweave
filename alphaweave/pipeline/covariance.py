"""Covariance shrinkage and robust estimation."""

import pandas as pd
import numpy as np
from typing import Literal


def shrink_cov_lw(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Ledoit-Wolf shrinkage to covariance matrix.

    Args:
        cov_matrix: Input covariance matrix

    Returns:
        Shrunk covariance matrix
    """
    S = cov_matrix.values
    n, p = S.shape

    if n != p:
        raise ValueError("Covariance matrix must be square")

    # Sample mean
    mu = np.trace(S) / p

    # Shrinkage target (scaled identity)
    F = mu * np.eye(p)

    # Compute shrinkage intensity
    # Simplified Ledoit-Wolf estimator
    sample_var = np.var(S.diagonal())
    target_var = np.var(F.diagonal())

    # Shrinkage coefficient (simplified)
    # In practice, use more sophisticated estimator
    delta = min(1.0, max(0.0, sample_var / (sample_var + target_var + 1e-10)))

    # Shrunk covariance
    S_shrunk = (1 - delta) * S + delta * F

    return pd.DataFrame(S_shrunk, index=cov_matrix.index, columns=cov_matrix.columns)


def shrink_cov_oas(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Oracle Approximating Shrinkage (OAS) to covariance matrix.

    Args:
        cov_matrix: Input covariance matrix

    Returns:
        Shrunk covariance matrix
    """
    S = cov_matrix.values
    n, p = S.shape

    if n != p:
        raise ValueError("Covariance matrix must be square")

    # Sample mean
    mu = np.trace(S) / p

    # Shrinkage target (scaled identity)
    F = mu * np.eye(p)

    # OAS shrinkage coefficient
    # More sophisticated than Ledoit-Wolf
    tr_S2 = np.trace(S @ S)
    tr_S = np.trace(S)
    tr_F2 = np.trace(F @ F)
    tr_F = np.trace(F)

    numerator = tr_S2 - (tr_S**2) / p
    denominator = (n + 1) * (tr_S2 - (tr_S**2) / p) + tr_F2 - (tr_F**2) / p

    if denominator > 0:
        rho = min(1.0, max(0.0, numerator / denominator))
    else:
        rho = 0.0

    # Shrunk covariance
    S_shrunk = (1 - rho) * S + rho * F

    return pd.DataFrame(S_shrunk, index=cov_matrix.index, columns=cov_matrix.columns)


def compute_factor_covariances(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute factor covariance matrix from factor returns.

    Args:
        factor_returns: DataFrame of factor returns [datetime × factors]

    Returns:
        Factor covariance matrix [factors × factors]
    """
    return factor_returns.cov()


def condition_number(cov_matrix: pd.DataFrame) -> float:
    """
    Compute condition number of covariance matrix.

    Args:
        cov_matrix: Covariance matrix

    Returns:
        Condition number
    """
    eigenvals = np.linalg.eigvals(cov_matrix.values)
    eigenvals = eigenvals[eigenvals > 0]  # Filter out numerical zeros
    if len(eigenvals) == 0:
        return np.inf
    return np.max(eigenvals) / np.min(eigenvals)


def is_positive_semidefinite(cov_matrix: pd.DataFrame, tol: float = 1e-8) -> bool:
    """
    Check if covariance matrix is positive semi-definite.

    Args:
        cov_matrix: Covariance matrix
        tol: Tolerance for eigenvalue check

    Returns:
        True if PSD, False otherwise
    """
    eigenvals = np.linalg.eigvals(cov_matrix.values)
    return np.all(eigenvals >= -tol)

