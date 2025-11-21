"""Tests for turnover helpers."""

import pandas as pd

from alphaweave.portfolio.turnover import (
    RebalancePenalty,
    TurnoverConstraint,
    apply_rebalance_penalty,
    apply_turnover_constraints,
    compute_turnover,
)


def test_compute_turnover_basic():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.2, "B": 0.8})
    turnover = compute_turnover(current, target)
    expected = abs(0.2 - 0.5) + abs(0.8 - 0.5)
    assert abs(turnover - expected) < 1e-12


def test_turnover_constraint_scales_trades():
    prev = pd.Series({"A": 0.0, "B": 0.0})
    target = pd.Series({"A": 1.0, "B": 0.0})
    constraint = TurnoverConstraint(max_turnover=0.2)
    adjusted = apply_turnover_constraints(prev, target, constraint)
    assert compute_turnover(prev, adjusted) <= 0.2 + 1e-9


def test_rebalance_penalty_weights_between_prev_and_target():
    prev = pd.Series({"A": 0.4, "B": 0.6})
    target = pd.Series({"A": 0.2, "B": 0.8})
    penalty = RebalancePenalty(lambda_rebalance=4.0)
    adjusted = apply_rebalance_penalty(prev, target, penalty)
    assert all(adjusted >= 0)
    # Weighted average between prev and target (alpha=1/(1+lambda)=0.2)
    expected = 0.2 * target + 0.8 * prev
    assert adjusted.equals(expected)


