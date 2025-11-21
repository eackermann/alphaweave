"""Paper trading adapter built on top of the mock portfolio broker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from alphaweave.core.types import Order
from alphaweave.live.adapters.base import BrokerAdapter, BrokerConfig
from alphaweave.live.adapters.mock import MockBrokerAdapter


@dataclass
class PaperBrokerAdapter(MockBrokerAdapter):
    """
    Paper trading adapter.

    Extends the mock adapter with a friendlier config surface so it can be used
    interchangeably with other adapters.
    """

    @classmethod
    def from_config(cls, config: BrokerConfig) -> "PaperBrokerAdapter":
        extra = config.extra or {}
        initial_cash = float(extra.get("initial_cash", 100_000.0))
        default_price = float(extra.get("default_price", 0.0))
        return cls(initial_cash=initial_cash, default_price=default_price, meta=extra)

    # Alias submit_order to clarify intention in docs
    def submit_order(self, order: Order) -> str:
        return super().submit_order(order)


