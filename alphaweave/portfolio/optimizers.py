"""Portfolio optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

from alphaweave.portfolio.constraints import PortfolioConstraints

Weights = pd.Series  # index = asset identifiers, values = weights


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: Weights
    objective_value: Optional[float] = None
    diagnostics: Optional[dict] = None

    def __repr__(self) -> str:
        return f"OptimizationResult(weights={len(self.weights)} assets, objective={self.objective_value})"


class PortfolioOptimizer(Protocol):
    """Protocol for portfolio optimizers."""

    def optimize(
        self,
        expected_returns: Optional[Weights] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        *,
        initial_weights: Optional[Weights] = None,
        constraints: Optional[Mapping] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns per asset (can be None for risk-only objectives)
            cov_matrix: Covariance matrix (required for risk-aware optimizers)
            initial_weights: Initial weights for optimization
            constraints: Portfolio constraints

        Returns:
            OptimizationResult with optimized weights
        """
        ...


def equal_weight(
    assets: Sequence[str],
    *,
    constraints: Optional[PortfolioConstraints] = None,
) -> OptimizationResult:
    """
    Equal weight portfolio optimizer.

    Args:
        assets: List of asset identifiers
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with equal weights
    """
    n = len(assets)
    if n == 0:
        raise ValueError("assets list cannot be empty")

    # Default: equal weights summing to 1.0
    target_gross = 1.0
    if constraints and constraints.target_gross is not None:
        target_gross = constraints.target_gross

    weight_value = target_gross / n

    # Apply bounds if specified
    if constraints:
        weights_dict = {}
        for asset in assets:
            lower, upper = constraints.get_bounds(asset)
            weights_dict[asset] = np.clip(weight_value, lower, upper)
        weights = pd.Series(weights_dict)
        # Renormalize to target_gross if needed
        current_sum = weights.sum()
        if abs(current_sum - target_gross) > 1e-6:
            weights = weights * (target_gross / current_sum)
    else:
        weights = pd.Series([weight_value] * n, index=assets)

    return OptimizationResult(weights=weights, objective_value=0.0)


def mean_variance(
    expected_returns: Weights,
    cov_matrix: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    max_weight: Optional[float] = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> OptimizationResult:
    """
    Mean-variance optimizer (maximize Sharpe ratio).

    Maximizes Sharpe ratio subject to:
    - sum(weights) = 1
    - long-only (weights >= 0)
    - Optional max_weight per asset

    Args:
        expected_returns: Expected returns per asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate for Sharpe calculation
        max_weight: Maximum weight per asset (default: None, no limit)
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with optimized weights

    Raises:
        ImportError: If scipy is not installed
    """
    if minimize is None:
        raise ImportError("scipy is required for mean_variance optimization. Install with: pip install scipy")
    """
    Mean-variance optimizer (maximize Sharpe ratio).

    Maximizes Sharpe ratio subject to:
    - sum(weights) = 1
    - long-only (weights >= 0)
    - Optional max_weight per asset

    Args:
        expected_returns: Expected returns per asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate for Sharpe calculation
        max_weight: Maximum weight per asset (default: None, no limit)
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with optimized weights
    """
    assets = list(expected_returns.index)
    n = len(assets)

    # Ensure cov_matrix is aligned with expected_returns
    cov_matrix = cov_matrix.loc[assets, assets]

    # Get bounds
    if constraints:
        lower_bounds, upper_bounds = constraints.get_bounds_arrays(assets)
    else:
        lower_bounds = [0.0] * n  # long-only default
        upper_bounds = [1.0] * n
        if max_weight is not None:
            upper_bounds = [min(1.0, max_weight)] * n

    # Objective: maximize Sharpe = (mu - rf) / sigma
    # Equivalently: minimize -Sharpe = -(mu - rf) / sigma
    mu = expected_returns.values
    rf = risk_free_rate

    def objective(w: np.ndarray) -> float:
        portfolio_return = np.dot(w, mu) - rf
        portfolio_var = np.dot(w, np.dot(cov_matrix.values, w))
        if portfolio_var <= 0:
            return 1e10  # Penalty for invalid variance
        portfolio_std = np.sqrt(portfolio_var)
        if portfolio_std < 1e-8:
            return 1e10
        sharpe = portfolio_return / portfolio_std
        return -sharpe  # Minimize negative Sharpe

    # Constraints: sum(weights) = 1
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Initial guess: equal weights
    x0 = np.ones(n) / n

    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower_bounds, upper_bounds)),
        constraints=[constraint_sum],
        options={"maxiter": 1000},
    )

    if not result.success:
        # Fallback to equal weights if optimization fails
        weights = pd.Series([1.0 / n] * n, index=assets)
        return OptimizationResult(weights=weights, objective_value=None)

    weights = pd.Series(result.x, index=assets)
    objective_value = -result.fun  # Convert back to positive Sharpe

    return OptimizationResult(weights=weights, objective_value=objective_value)


def min_variance(
    cov_matrix: pd.DataFrame,
    *,
    max_weight: Optional[float] = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> OptimizationResult:
    """
    Minimum variance optimizer.

    Minimizes portfolio variance subject to:
    - sum(weights) = 1
    - long-only (weights >= 0)
    - Optional max_weight per asset

    Args:
        cov_matrix: Covariance matrix
        max_weight: Maximum weight per asset (default: None, no limit)
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with optimized weights

    Raises:
        ImportError: If scipy is not installed
    """
    if minimize is None:
        raise ImportError("scipy is required for min_variance optimization. Install with: pip install scipy")
    """
    Minimum variance optimizer.

    Minimizes portfolio variance subject to:
    - sum(weights) = 1
    - long-only (weights >= 0)
    - Optional max_weight per asset

    Args:
        cov_matrix: Covariance matrix
        max_weight: Maximum weight per asset (default: None, no limit)
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with optimized weights
    """
    assets = list(cov_matrix.index)
    n = len(assets)

    # Get bounds
    if constraints:
        lower_bounds, upper_bounds = constraints.get_bounds_arrays(assets)
    else:
        lower_bounds = [0.0] * n  # long-only default
        upper_bounds = [1.0] * n
        if max_weight is not None:
            upper_bounds = [min(1.0, max_weight)] * n

    # Objective: minimize portfolio variance = w^T * Σ * w
    def objective(w: np.ndarray) -> float:
        return np.dot(w, np.dot(cov_matrix.values, w))

    # Constraints: sum(weights) = 1
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Initial guess: equal weights
    x0 = np.ones(n) / n

    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower_bounds, upper_bounds)),
        constraints=[constraint_sum],
        options={"maxiter": 1000},
    )

    if not result.success:
        # Fallback to equal weights if optimization fails
        weights = pd.Series([1.0 / n] * n, index=assets)
        return OptimizationResult(weights=weights, objective_value=None)

    weights = pd.Series(result.x, index=assets)
    portfolio_var = result.fun

    return OptimizationResult(weights=weights, objective_value=portfolio_var)


def risk_parity(
    cov_matrix: pd.DataFrame,
    *,
    constraints: Optional[PortfolioConstraints] = None,
) -> OptimizationResult:
    """
    Risk parity optimizer (equal risk contribution).

    Computes weights so that each asset contributes equally to total portfolio risk.

    Args:
        cov_matrix: Covariance matrix
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with risk parity weights

    Raises:
        ImportError: If scipy is not installed
    """
    if minimize is None:
        raise ImportError("scipy is required for risk_parity optimization. Install with: pip install scipy")
    """
    Risk parity optimizer (equal risk contribution).

    Computes weights so that each asset contributes equally to total portfolio risk.

    Args:
        cov_matrix: Covariance matrix
        constraints: Optional portfolio constraints

    Returns:
        OptimizationResult with risk parity weights
    """
    assets = list(cov_matrix.index)
    n = len(assets)

    # Get bounds
    if constraints:
        lower_bounds, upper_bounds = constraints.get_bounds_arrays(assets)
    else:
        lower_bounds = [0.0] * n  # long-only default
        upper_bounds = [1.0] * n

    # Risk contribution of asset i: RC_i = w_i * (Σw)_i
    # Objective: minimize sum over i (RC_i - RC_target)²
    # where RC_target = 1/n (equal risk contribution)

    cov = cov_matrix.values

    def risk_contributions(w: np.ndarray) -> np.ndarray:
        """Compute risk contributions for each asset."""
        portfolio_std = np.sqrt(np.dot(w, np.dot(cov, w)))
        if portfolio_std < 1e-8:
            return np.ones(n) / n
        marginal_contrib = np.dot(cov, w) / portfolio_std
        return w * marginal_contrib

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w)
        target_rc = np.ones(n) / n  # Equal risk contribution
        return np.sum((rc - target_rc) ** 2)

    # Constraints: sum(weights) = 1
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Initial guess: equal weights
    x0 = np.ones(n) / n

    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower_bounds, upper_bounds)),
        constraints=[constraint_sum],
        options={"maxiter": 1000},
    )

    if not result.success:
        # Fallback to equal weights if optimization fails
        weights = pd.Series([1.0 / n] * n, index=assets)
        return OptimizationResult(weights=weights, objective_value=None)

    weights = pd.Series(result.x, index=assets)
    return OptimizationResult(weights=weights, objective_value=result.fun)


def target_volatility(
    base_weights: Weights,
    cov_matrix: pd.DataFrame,
    *,
    target_vol: float,
    max_leverage: Optional[float] = None,
) -> OptimizationResult:
    """
    Target volatility allocator.

    Scales base weights so that portfolio volatility matches target.

    Args:
        base_weights: Base portfolio weights
        cov_matrix: Covariance matrix
        target_vol: Target annualized volatility (e.g., 0.10 for 10%)
        max_leverage: Maximum leverage (sum of absolute weights)

    Returns:
        OptimizationResult with scaled weights
    """
    assets = list(base_weights.index)

    # Ensure cov_matrix is aligned
    cov_matrix = cov_matrix.loc[assets, assets]

    # Compute current portfolio volatility
    w = base_weights.values
    portfolio_var = np.dot(w, np.dot(cov_matrix.values, w))
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol < 1e-8:
        # If base portfolio has zero volatility, return as-is
        return OptimizationResult(weights=base_weights, objective_value=0.0)

    # Scale factor
    scale = target_vol / portfolio_vol

    # Apply leverage constraint if specified
    if max_leverage is not None:
        current_gross = np.sum(np.abs(base_weights))
        max_scale = max_leverage / current_gross if current_gross > 0 else 1.0
        scale = min(scale, max_scale)

    scaled_weights = base_weights * scale

    return OptimizationResult(weights=scaled_weights, objective_value=abs(portfolio_vol - target_vol))

