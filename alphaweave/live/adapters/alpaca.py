"""Alpaca adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from alphaweave.core.types import Order
from alphaweave.live.adapters.base import BrokerAdapter, BrokerConfig
from alphaweave.live.broker import BrokerAccountState, BrokerFill


@dataclass
class AlpacaAdapter(BrokerAdapter):
    api_key: str
    api_secret: str
    base_url: str
    account_id: str | None = None
    extra: Mapping[str, Any] | None = None

    def connect(self) -> None:
        raise NotImplementedError("AlpacaAdapter.connect must establish REST/WebSocket sessions")

    def disconnect(self) -> None:
        raise NotImplementedError

    def submit_order(self, order: Order) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    def get_account_state(self) -> BrokerAccountState:
        raise NotImplementedError

    def poll_fills(self) -> Sequence[BrokerFill]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BrokerConfig) -> "AlpacaAdapter":
        extra = config.extra or {}
        return cls(
            api_key=str(extra["api_key"]),
            api_secret=str(extra["api_secret"]),
            base_url=str(extra.get("base_url", "https://paper-api.alpaca.markets")),
            account_id=config.account_id,
            extra=extra,
        )


