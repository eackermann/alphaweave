"""Core datatypes for alphaweave."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional


@dataclass
class Bar:
    """OHLCV bar data."""

    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    symbol: Optional[str] = None

    def __repr__(self) -> str:
        vol_str = f", volume={self.volume}" if self.volume is not None else ""
        sym_str = f", symbol={self.symbol}" if self.symbol is not None else ""
        return (
            f"Bar(datetime={self.datetime}, open={self.open}, high={self.high}, "
            f"low={self.low}, close={self.close}{vol_str}{sym_str})"
        )


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TARGET_PERCENT = auto()


@dataclass
class Order:
    """Order request."""

    symbol: str
    size: float  # positive = buy, negative = sell
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_percent: Optional[float] = None

    def __repr__(self) -> str:
        parts = [
            f"symbol={self.symbol}",
            f"size={self.size}",
            f"order_type={self.order_type.name}",
        ]
        if self.limit_price is not None:
            parts.append(f"limit={self.limit_price}")
        if self.stop_price is not None:
            parts.append(f"stop={self.stop_price}")
        if self.target_percent is not None:
            parts.append(f"target={self.target_percent}")
        return f"Order({', '.join(parts)})"


@dataclass
class Fill:
    """Executed order fill."""

    order_id: int
    symbol: str
    size: float
    price: float
    datetime: datetime

    def __repr__(self) -> str:
        return (
            f"Fill(order_id={self.order_id}, symbol={self.symbol}, size={self.size}, "
            f"price={self.price}, datetime={self.datetime})"
        )


@dataclass
class Position:
    """Portfolio position."""

    symbol: str
    size: float
    avg_price: float

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol}, size={self.size}, avg_price={self.avg_price})"
        )

