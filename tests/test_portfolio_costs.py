"""Tests for transaction cost models."""

import pandas as pd

from alphaweave.portfolio.costs import (
    CompositeCostModel,
    ImpactCostModel,
    ProportionalCostModel,
    SpreadBasedCostModel,
)


def test_proportional_cost_model_scalar():
    current = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.3, "B": 0.7})
    model = ProportionalCostModel(cost_per_dollar=0.001)
    expected = (target - current).abs().sum() * 0.001
    assert model.estimate_cost(current, target) == expected


def test_proportional_cost_model_series():
    current = pd.Series({"A": 0.0, "B": 0.0})
    target = pd.Series({"A": 0.8, "B": 0.2})
    model = ProportionalCostModel(cost_per_dollar=pd.Series({"A": 0.002, "B": 0.001}))
    expected = (target["A"] * 0.002) + (target["B"] * 0.001)
    assert abs(model.estimate_cost(current, target) - expected) < 1e-12


def test_spread_based_cost_model():
    current = pd.Series({"A": 0.0, "B": 0.0})
    target = pd.Series({"A": 0.5, "B": 0.5})
    model = SpreadBasedCostModel(spread_bps=10, participation_rate=0.5)
    turnover = (target - current).abs().sum()
    expected = turnover * 0.5 * (10 / 10000) * 0.5
    assert abs(model.estimate_cost(current, target) - expected) < 1e-12


def test_impact_cost_model_scaling():
    current = pd.Series({"A": 0.0, "B": 0.0})
    target_small = pd.Series({"A": 0.1, "B": 0.1})
    target_large = pd.Series({"A": 0.4, "B": 0.4})
    adv = pd.Series({"A": 1.0, "B": 1.0})
    model = ImpactCostModel(adv=adv, impact_coeff=0.2)

    cost_small = model.estimate_cost(current, target_small)
    cost_large = model.estimate_cost(current, target_large)
    assert cost_large > cost_small


def test_composite_cost_model_sum():
    current = pd.Series({"A": 0.2, "B": 0.8})
    target = pd.Series({"A": 0.4, "B": 0.6})
    model = CompositeCostModel(
        components=[
            ProportionalCostModel(cost_per_dollar=0.001),
            SpreadBasedCostModel(spread_bps=5),
        ]
    )
    cost_single = model.components[0].estimate_cost(current, target) + model.components[1].estimate_cost(current, target)
    assert abs(model.estimate_cost(current, target) - cost_single) < 1e-12


