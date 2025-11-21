"""Portfolio constraint definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class WeightBounds:
    """Bounds for individual asset weights."""

    lower: float  # e.g. 0.0 for long-only
    upper: float  # e.g. 0.1 for 10% cap


@dataclass
class PortfolioConstraints:
    """
    Portfolio-level constraints for optimization.

    Attributes:
        weight_bounds: Per-asset weight bounds (asset -> WeightBounds)
        default_bounds: Fallback bounds for assets not in weight_bounds
        target_gross: Target gross exposure (sum of absolute weights)
        long_only: If True, enforce long-only constraint (weights >= 0)
    """

    weight_bounds: Optional[Mapping[str, WeightBounds]] = None  # per asset
    default_bounds: Optional[WeightBounds] = None  # fallback
    target_gross: Optional[float] = 1.0  # sum(|w|) constraint
    long_only: bool = True

    def get_bounds(self, asset: str) -> tuple[float, float]:
        """
        Get lower and upper bounds for an asset.

        Args:
            asset: Asset identifier

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.weight_bounds and asset in self.weight_bounds:
            bounds = self.weight_bounds[asset]
            return (bounds.lower, bounds.upper)

        if self.default_bounds:
            return (self.default_bounds.lower, self.default_bounds.upper)

        # Default: long-only with no upper bound
        if self.long_only:
            return (0.0, 1.0)
        return (-1.0, 1.0)

    def get_bounds_arrays(self, assets: list[str]) -> tuple[list[float], list[float]]:
        """
        Get lower and upper bound arrays for a list of assets.

        Args:
            assets: List of asset identifiers

        Returns:
            Tuple of (lower_bounds, upper_bounds) lists
        """
        lower = []
        upper = []
        for asset in assets:
            lb, ub = self.get_bounds(asset)
            lower.append(lb)
            upper.append(ub)
        return (lower, upper)

