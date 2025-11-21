"""Slippage models for execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from alphaweave.core.types import Order


class SlippageModel(ABC):
    """Interface for slippage calculations."""

    @abstractmethod
    def execute(self, order: Order, bar) -> float:
        """Return execution price for this order given the bar data."""


class NoSlippage(SlippageModel):
    """Executes exactly at the provided bar close."""

    def execute(self, order: Order, bar) -> float:
        return float(bar.close)


class FixedBpsSlippage(SlippageModel):
    """Applies a fixed basis-point adjustment to execution price."""

    def __init__(self, bps: float):
        self.bps = bps

    def execute(self, order: Order, bar) -> float:
        sign = 1.0 if order.size > 0 else -1.0
        return float(bar.close * (1.0 + sign * self.bps / 10000.0))
