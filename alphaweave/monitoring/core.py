"""Core monitoring primitives for alphaweave."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, List, Mapping, Protocol, Sequence

import pandas as pd


@dataclass
class BarSnapshot:
    """State captured for each processed bar."""

    timestamp: datetime
    prices: Mapping[str, float]
    equity: float
    cash: float
    positions_value: float
    positions: Mapping[str, float]
    leverage: float | None = None
    pnl: float | None = None
    drawdown: float | None = None


@dataclass
class TradeRecord:
    """Executed trade information suitable for monitoring."""

    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    fees: float
    slippage: float | None = None
    order_id: str | None = None
    strategy_tag: str | None = None
    pnl: float | None = None


class Monitor(Protocol):
    """Protocol for monitoring hooks provided to engines."""

    def on_run_start(self, meta: Mapping[str, Any] | None = None) -> None:
        ...

    def on_bar(self, snapshot: BarSnapshot) -> None:
        ...

    def on_trade(self, trade: TradeRecord) -> None:
        ...

    def on_metric(self, name: str, value: float, timestamp: datetime) -> None:
        ...

    def on_run_end(self) -> None:
        ...


@dataclass
class InMemoryMonitor:
    """Simple monitor that stores everything in memory (pandas-friendly)."""

    bars: List[BarSnapshot] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    metrics: List[tuple[str, datetime, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    _ended: bool = False

    def on_run_start(self, meta: Mapping[str, Any] | None = None) -> None:
        self.meta = dict(meta or {})
        self._ended = False

    def on_bar(self, snapshot: BarSnapshot) -> None:
        self.bars.append(snapshot)

    def on_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)

    def on_metric(self, name: str, value: float, timestamp: datetime) -> None:
        self.metrics.append((name, timestamp, float(value)))

    def on_run_end(self) -> None:
        self._ended = True

    # DataFrame helpers ---------------------------------------------------
    def bars_df(self) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "equity",
                    "cash",
                    "positions_value",
                    "leverage",
                    "pnl",
                    "drawdown",
                    "prices",
                    "positions",
                ]
            )
        records = []
        for snapshot in self.bars:
            record = {
                "timestamp": snapshot.timestamp,
                "equity": snapshot.equity,
                "cash": snapshot.cash,
                "positions_value": snapshot.positions_value,
                "leverage": snapshot.leverage,
                "pnl": snapshot.pnl,
                "drawdown": snapshot.drawdown,
                "prices": dict(snapshot.prices),
                "positions": dict(snapshot.positions),
            }
            records.append(record)
        return pd.DataFrame(records)

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "price",
                    "size",
                    "fees",
                    "slippage",
                    "order_id",
                    "strategy_tag",
                    "pnl",
                ]
            )
        records = []
        for trade in self.trades:
            record = asdict(trade)
            records.append(record)
        return pd.DataFrame(records)

    def metrics_df(self) -> pd.DataFrame:
        if not self.metrics:
            return pd.DataFrame(columns=["name", "timestamp", "value"])
        return pd.DataFrame(
            [
                {"name": name, "timestamp": ts, "value": value}
                for name, ts, value in self.metrics
            ]
        )


