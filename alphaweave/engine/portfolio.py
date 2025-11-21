"""Portfolio accounting for the vector backtester."""

from __future__ import annotations

from typing import Dict

from alphaweave.core.types import Position, Fill


class Portfolio:
    """Tracks cash and positions for a backtest run."""

    def __init__(self, starting_cash: float):
        self.cash: float = starting_cash
        self.positions: Dict[str, Position] = {}

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions given a trade fill."""
        cost = fill.size * fill.price
        self.cash -= cost

        position = self.positions.get(fill.symbol)
        if position is None:
            position = Position(symbol=fill.symbol, size=0.0, avg_price=0.0)

        new_size = position.size + fill.size

        if fill.size > 0:  # Buying, update average price
            total_cost = position.size * position.avg_price + fill.size * fill.price
            avg_price = total_cost / new_size if new_size != 0 else fill.price
        else:  # Selling, keep previous avg_price unless position closed
            avg_price = position.avg_price

        if abs(new_size) < 1e-10:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                size=new_size,
                avg_price=avg_price,
            )

    def update_price(self, symbol: str, price: float) -> None:  # pragma: no cover - placeholder
        """Placeholder for future mark-to-market logic."""
        return

    def position_value(self, symbol: str, price: float) -> float:
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        return position.size * price

    def total_value(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            total += position.size * price
        return total
