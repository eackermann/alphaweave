"""Turnover helpers and constraints for portfolio rebalancing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_turnover(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    Compute turnover as sum of absolute weight changes.

    This corresponds to the fraction of portfolio value traded when moving
    from current_weights to target_weights.
    """
    aligned_index = target_weights.index.union(current_weights.index)
    current = current_weights.reindex(aligned_index).fillna(0.0)
    target = target_weights.reindex(aligned_index).fillna(0.0)
    delta = (target - current).abs()
    return float(delta.sum())


@dataclass
class TurnoverConstraint:
    """
    Configuration for hard turnover limits.

    max_turnover: total turnover allowed per rebalance (e.g., 0.2 = 20%).
    max_change_per_asset: per-asset absolute weight change cap.
    """

    max_turnover: float | None = None
    max_change_per_asset: float | None = None


@dataclass
class RebalancePenalty:
    """Soft turnover control via squared deviation from previous weights."""

    lambda_rebalance: float = 1.0


def apply_turnover_constraints(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    constraint: TurnoverConstraint | None,
) -> pd.Series:
    """Apply turnover constraints using shrink-and-clip heuristics."""
    if constraint is None:
        return target_weights

    adjusted = target_weights.copy()
    prev = prev_weights.reindex(adjusted.index).fillna(0.0)

    # Enforce per-asset change cap first
    if constraint.max_change_per_asset is not None:
        delta = (adjusted - prev).clip(
            lower=-constraint.max_change_per_asset,
            upper=constraint.max_change_per_asset,
        )
        adjusted = prev + delta

    # Enforce total turnover cap by scaling trade vector
    if constraint.max_turnover is not None and constraint.max_turnover >= 0:
        delta = adjusted - prev
        turnover = float(delta.abs().sum())
        if turnover > constraint.max_turnover and turnover > 0:
            scale = constraint.max_turnover / turnover
            scale = np.clip(scale, 0.0, 1.0)
            adjusted = prev + delta * scale

    return adjusted


def apply_rebalance_penalty(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    penalty: RebalancePenalty | None,
) -> pd.Series:
    """Interpolate between previous and target weights based on penalty."""

    if penalty is None:
        return target_weights

    lam = max(penalty.lambda_rebalance, 0.0)
    if lam == 0:
        return target_weights

    prev = prev_weights.reindex(target_weights.index).fillna(0.0)
    alpha = 1.0 / (1.0 + lam)
    adjusted = alpha * target_weights + (1.0 - alpha) * prev
    return adjusted


