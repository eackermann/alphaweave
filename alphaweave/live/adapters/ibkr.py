"""IBKR adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from alphaweave.core.types import Order
from alphaweave.live.adapters.base import BrokerAdapter, BrokerConfig
from alphaweave.live.broker import BrokerAccountState, BrokerFill


@dataclass
class IBKRAdapter(BrokerAdapter):
    host: str
    port: int
    client_id: int
    account_id: str
    extra: Mapping[str, Any] | None = None

    def connect(self) -> None:
        raise NotImplementedError("IBKRAdapter.connect is broker-specific")

    def disconnect(self) -> None:
        raise NotImplementedError("IBKRAdapter.disconnect is broker-specific")

    def submit_order(self, order: Order) -> str:
        raise NotImplementedError("IBKRAdapter.submit_order must call TWS/IB Gateway APIs")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    def get_account_state(self) -> BrokerAccountState:
        raise NotImplementedError

    def poll_fills(self) -> Sequence[BrokerFill]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BrokerConfig) -> "IBKRAdapter":
        extra = config.extra or {}
        return cls(
            host=str(extra["host"]),
            port=int(extra.get("port", 7497)),
            client_id=int(extra.get("client_id", 1)),
            account_id=config.account_id or "",
            extra=extra,
        )


