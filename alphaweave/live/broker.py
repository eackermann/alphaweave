"""Broker protocol definitions for live trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol, Sequence

from alphaweave.core.types import Order


@dataclass
class BrokerFill:
    """Represents an execution reported by the broker."""

    order_id: str
    symbol: str
    size: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    slippage: float | None = None


@dataclass
class BrokerAccountState:
    """Snapshot of broker-reported balances and positions."""

    timestamp: datetime
    equity: float
    cash: float
    positions: Mapping[str, float]
    currency: str | None = None


class Broker(Protocol):
    """Protocol that all broker integrations must satisfy."""

    def submit_order(self, order: Order) -> str:
        """Submit an order and return broker-side order id."""

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""

    def get_open_orders(self) -> Sequence[Mapping[str, float | str]]:
        """Return a lightweight description of working orders."""

    def get_account_state(self) -> BrokerAccountState:
        """Return most recent account snapshot."""

    def poll_fills(self) -> Sequence[BrokerFill]:
        """Return any new fills since last poll."""


