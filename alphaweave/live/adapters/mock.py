"""In-process mock broker adapter for tests and dry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from alphaweave.core.types import Fill, Order
from alphaweave.engine.portfolio import Portfolio
from alphaweave.live.adapters.base import BrokerAdapter, BrokerConfig
from alphaweave.live.broker import BrokerAccountState, BrokerFill


@dataclass
class MockBrokerAdapter(BrokerAdapter):
    """Simple broker adapter that mirrors fills locally."""

    initial_cash: float = 100_000.0
    default_price: float = 0.0
    meta: Mapping[str, Any] | None = None

    _connected: bool = field(init=False, default=False)
    _portfolio: Portfolio = field(init=False)
    _pending_fills: list[BrokerFill] = field(init=False, default_factory=list)
    _order_seq: int = field(init=False, default=0)
    _last_prices: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._portfolio = Portfolio(self.initial_cash)

    @classmethod
    def from_config(cls, config: BrokerConfig) -> "MockBrokerAdapter":
        extra = config.extra or {}
        return cls(
            initial_cash=float(extra.get("initial_cash", 100_000.0)),
            default_price=float(extra.get("default_price", 0.0)),
            meta=config.extra,
        )

    # BrokerAdapter lifecycle -------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # Broker interface --------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        self._ensure_connected()
        order_id = str(uuid4())
        price = self._infer_price(order)
        timestamp = datetime.now(timezone.utc)
        size = float(order.size)

        self._order_seq += 1
        fill = Fill(order_id=self._order_seq, symbol=order.symbol, size=size, price=price, datetime=timestamp)
        self._portfolio.apply_fill(fill)

        broker_fill = BrokerFill(
            order_id=order_id,
            symbol=order.symbol,
            size=size,
            price=price,
            timestamp=timestamp,
            fees=0.0,
        )
        self._pending_fills.append(broker_fill)
        return order_id

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - nothing to cancel
        return

    def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        return []

    def get_account_state(self) -> BrokerAccountState:
        timestamp = datetime.now(timezone.utc)
        equity = self._portfolio.total_value(self._last_prices)
        positions = {sym: pos.size for sym, pos in self._portfolio.positions.items()}
        return BrokerAccountState(
            timestamp=timestamp,
            equity=equity,
            cash=self._portfolio.cash,
            positions=positions,
        )

    def poll_fills(self) -> Sequence[BrokerFill]:
        fills = list(self._pending_fills)
        self._pending_fills.clear()
        return fills

    # Utilities ----------------------------------------------------------------
    def update_prices(self, prices: Mapping[str, float]) -> None:
        """Update internal marks so account equity reflects latest data."""
        self._last_prices.update(prices)

    def _infer_price(self, order: Order) -> float:
        if order.limit_price is not None:
            return float(order.limit_price)
        if order.stop_price is not None:
            return float(order.stop_price)
        return float(self._last_prices.get(order.symbol, self.default_price))

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MockBrokerAdapter not connected")


