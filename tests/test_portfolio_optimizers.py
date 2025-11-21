"""Tests for portfolio optimizers."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from alphaweave.portfolio.constraints import PortfolioConstraints, WeightBounds
from alphaweave.portfolio.costs import ProportionalCostModel
from alphaweave.portfolio.optimizers import (
    OptimizationResult,
    equal_weight,
    mean_variance,
    min_variance,
    risk_parity,
    target_volatility,
)
from alphaweave.portfolio.turnover import RebalancePenalty, TurnoverConstraint, compute_turnover


def test_equal_weight():
    """Test equal weight optimizer."""
    assets = ["A", "B", "C"]
    result = equal_weight(assets)

    assert isinstance(result, OptimizationResult)
    assert len(result.weights) == 3
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert all(abs(w - 1.0 / 3) < 1e-6 for w in result.weights.values)


def test_equal_weight_with_constraints():
    """Test equal weight with constraints."""
    assets = ["A", "B", "C"]
    constraints = PortfolioConstraints(
        default_bounds=WeightBounds(lower=0.0, upper=0.4),
        target_gross=1.0,
    )
    result = equal_weight(assets, constraints=constraints)

    assert len(result.weights) == 3
    assert all(w <= 0.4 for w in result.weights.values)
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_min_variance():
    """Test minimum variance optimizer."""
    # Create simple covariance matrix
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01, 0.01],
            "B": [0.01, 0.04, 0.01],
            "C": [0.01, 0.01, 0.04],
        },
        index=["A", "B", "C"],
    )

    result = min_variance(cov)

    assert isinstance(result, OptimizationResult)
    assert len(result.weights) == 3
    assert abs(result.weights.sum() - 1.0) < 1e-3
    assert all(w >= 0 for w in result.weights.values)  # Long-only


def test_mean_variance():
    """Test mean-variance optimizer."""
    # Create expected returns and covariance
    expected_returns = pd.Series({"A": 0.10, "B": 0.08, "C": 0.06})
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01, 0.01],
            "B": [0.01, 0.04, 0.01],
            "C": [0.01, 0.01, 0.04],
        },
        index=["A", "B", "C"],
    )

    result = mean_variance(expected_returns, cov)

    assert isinstance(result, OptimizationResult)
    assert len(result.weights) == 3
    assert abs(result.weights.sum() - 1.0) < 1e-3
    assert all(w >= 0 for w in result.weights.values)  # Long-only
    # Asset A should have higher weight (higher expected return)
    assert result.weights["A"] >= result.weights["C"]


def test_risk_parity():
    """Test risk parity optimizer."""
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01, 0.01],
            "B": [0.01, 0.04, 0.01],
            "C": [0.01, 0.01, 0.04],
        },
        index=["A", "B", "C"],
    )

    result = risk_parity(cov)

    assert isinstance(result, OptimizationResult)
    assert len(result.weights) == 3
    assert abs(result.weights.sum() - 1.0) < 1e-3
    assert all(w >= 0 for w in result.weights.values)


def test_mean_variance_turnover_constraint():
    expected_returns = pd.Series({"A": 0.10, "B": 0.09, "C": 0.08})
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01, 0.01],
            "B": [0.01, 0.04, 0.01],
            "C": [0.01, 0.01, 0.04],
        },
        index=["A", "B", "C"],
    )
    prev = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    constraint = TurnoverConstraint(max_turnover=0.2)
    result = mean_variance(
        expected_returns,
        cov,
        prev_weights=prev,
        turnover_constraint=constraint,
    )
    turnover = compute_turnover(prev, result.weights)
    assert turnover <= 0.2 + 1e-6


def test_mean_variance_cost_penalty_shrinks_change():
    expected_returns = pd.Series({"A": 0.12, "B": 0.08})
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.0],
            "B": [0.0, 0.04],
        },
        index=["A", "B"],
    )
    prev = pd.Series({"A": 0.0, "B": 1.0})

    result_no_cost = mean_variance(expected_returns, cov, prev_weights=prev)
    model = ProportionalCostModel(cost_per_dollar=10.0)
    result_with_cost = mean_variance(
        expected_returns,
        cov,
        prev_weights=prev,
        transaction_cost_model=model,
    )
    # Cost penalty should keep more weight in previous allocation (asset B)
    assert result_with_cost.weights["B"] > result_no_cost.weights["B"]


def test_rebalance_penalty_interpolates_weights():
    expected_returns = pd.Series({"A": 0.11, "B": 0.10})
    cov = pd.DataFrame(
        {
            "A": [0.02, 0.0],
            "B": [0.0, 0.03],
        },
        index=["A", "B"],
    )
    prev = pd.Series({"A": 0.7, "B": 0.3})
    base = mean_variance(expected_returns, cov, prev_weights=prev)
    penalty = RebalancePenalty(lambda_rebalance=4.0)
    result = mean_variance(
        expected_returns,
        cov,
        prev_weights=prev,
        rebalance_penalty=penalty,
    )
    alpha = 1.0 / (1.0 + penalty.lambda_rebalance)
    expected = alpha * base.weights + (1 - alpha) * prev
    pdt.assert_series_equal(result.weights, expected)


def test_target_volatility():
    """Test target volatility allocator."""
    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01],
            "B": [0.01, 0.04],
        },
        index=["A", "B"],
    )

    result = target_volatility(base_weights, cov, target_vol=0.10)

    assert isinstance(result, OptimizationResult)
    assert len(result.weights) == 2
    # Weights should be scaled
    assert abs(result.weights.sum() - 1.0) < 1.0  # May be scaled up/down


def test_target_volatility_with_max_leverage():
    """Test target volatility with leverage constraint."""
    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    cov = pd.DataFrame(
        {
            "A": [0.04, 0.01],
            "B": [0.01, 0.04],
        },
        index=["A", "B"],
    )

    result = target_volatility(base_weights, cov, target_vol=0.20, max_leverage=1.5)

    # Gross exposure should be <= max_leverage
    gross_exposure = result.weights.abs().sum()
    assert gross_exposure <= 1.5 + 1e-6

