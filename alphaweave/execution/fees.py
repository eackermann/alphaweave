"""Fee models for execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from alphaweave.core.types import Order, Fill


class FeesModel(ABC):
    """Interface for calculating transaction fees."""

    @abstractmethod
    def calculate(self, order: Order, fill: Fill) -> float:
        """Return the positive fee cost for a fill."""


class NoFees(FeesModel):
    """Zero fees model."""

    def calculate(self, order: Order, fill: Fill) -> float:
        return 0.0


class PerShareFees(FeesModel):
    """Per-share flat rate fees."""

    def __init__(self, rate: float):
        self.rate = rate

    def calculate(self, order: Order, fill: Fill) -> float:
        return abs(fill.size) * self.rate


class PercentageFees(FeesModel):
    """Fees as a percentage of notional value."""

    def __init__(self, rate: float):
        self.rate = rate

    def calculate(self, order: Order, fill: Fill) -> float:
        return abs(fill.size * fill.price) * self.rate
