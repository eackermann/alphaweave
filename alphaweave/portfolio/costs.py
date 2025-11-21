"""Transaction cost models at the portfolio (weight) level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def _align_weights(
    current_weights: pd.Series | None,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align weight series to a common index, filling missing values with 0."""
    if current_weights is None:
        current_weights = pd.Series(0.0, index=target_weights.index)
    aligned_index = target_weights.index.union(current_weights.index)
    current = current_weights.reindex(aligned_index).fillna(0.0)
    target = target_weights.reindex(aligned_index).fillna(0.0)
    return current, target


def _to_series(value: pd.Series | float, index: Sequence[str], name: str) -> pd.Series:
    """Convert scalar/series into Series aligned to index."""
    if isinstance(value, pd.Series):
        series = value.reindex(index).ffill().bfill().fillna(0.0)
        return series
    return pd.Series(float(value), index=index, name=name)


@dataclass
class TransactionCostModel:
    """Interface for portfolio-level transaction cost models."""

    def estimate_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> float:
        """
        Estimate expected transaction cost as a fraction of portfolio value.

        Returns:
            Cost expressed in decimal terms (e.g. 0.001 = 10 bps).
        """
        raise NotImplementedError


@dataclass
class ProportionalCostModel(TransactionCostModel):
    """
    Linear cost model proportional to turnover.

    cost_per_dollar defines the fraction of portfolio value paid per unit
    of turnover (sum abs weight change).
    """

    cost_per_dollar: pd.Series | float

    def estimate_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> float:
        current, target = _align_weights(current_weights, target_weights)
        delta = (target - current).abs()
        costs = _to_series(self.cost_per_dollar, delta.index, "cost_per_dollar")
        return float((delta * costs).sum())


@dataclass
class SpreadBasedCostModel(TransactionCostModel):
    """
    Cost model based on bid/ask spread estimates.

    spread_bps represents the half-spread in basis points. The total cost
    scales with the traded notional and optional participation rate.
    """

    spread_bps: pd.Series | float
    participation_rate: float = 1.0

    def estimate_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> float:
        current, target = _align_weights(current_weights, target_weights)
        delta = (target - current).abs()
        spread_fraction = _to_series(self.spread_bps, delta.index, "spread_bps") / 10000.0
        effective_cost = 0.5 * spread_fraction * self.participation_rate
        return float((delta * effective_cost).sum())


@dataclass
class ImpactCostModel(TransactionCostModel):
    """
    Simple square-root market impact model.

    adv should represent the average daily dollar volume relative to the
    portfolio value (e.g., ADV / portfolio_value). Higher ADV implies lower
    impact for a given trade size.
    """

    adv: pd.Series
    impact_coeff: float = 0.1

    def estimate_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> float:
        if (self.adv <= 0).any():
            raise ValueError("Average daily volume (adv) must be positive for all assets")

        current, target = _align_weights(current_weights, target_weights)
        delta = (target - current).abs()
        adv = self.adv.reindex(delta.index).ffill().bfill()
        adv = adv.replace(0, np.nan)
        volume_fraction = delta / adv
        volume_fraction = volume_fraction.clip(lower=0.0).fillna(0.0)
        impact = self.impact_coeff * np.sqrt(volume_fraction)
        return float((impact * delta).sum())


@dataclass
class CompositeCostModel(TransactionCostModel):
    """Combine multiple cost models by summing their estimates."""

    components: list[TransactionCostModel]

    def estimate_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> float:
        if not self.components:
            return 0.0
        return float(
            sum(
                component.estimate_cost(
                    current_weights=current_weights,
                    target_weights=target_weights,
                    prices=prices,
                    volumes=volumes,
                )
                for component in self.components
            )
        )


