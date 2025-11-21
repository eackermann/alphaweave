"""
Example: Strategy that trades around earnings events.

This strategy demonstrates how to use EventStore and event helpers
to react to earnings announcements.
"""

import pandas as pd
from datetime import datetime, timezone

from alphaweave.core.frame import Frame
from alphaweave.data.events import Event, EventStore, load_events_csv
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class EarningsPlayStrategy(Strategy):
    """Trade around earnings announcements."""

    def init(self):
        self.lookback = "3D"  # 3-day window to check for earnings
        self.hold_days = 2  # Hold for 2 days after entry
        self.entry_record: dict[str, int] = {}  # Track entry bar for each symbol

    def next(self, i):
        symbols = ["AAPL", "MSFT", "GOOG"]

        for symbol in symbols:
            # Check if earnings occurred in the last 3 days
            if self.has_event(type="earnings", symbol=symbol, window=self.lookback):
                # Enter position
                if symbol not in self.entry_record:
                    self.order_target_percent(symbol, 1.0)
                    self.entry_record[symbol] = i

            # Exit after holding for specified days
            if symbol in self.entry_record:
                bars_held = i - self.entry_record[symbol]
                if bars_held >= self.hold_days:
                    self.order_target_percent(symbol, 0.0)
                    del self.entry_record[symbol]


def main():
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    data = {
        "AAPL": Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [150.0] * 30,
                    "high": [151.0] * 30,
                    "low": [149.0] * 30,
                    "close": [150.5] * 30,
                    "volume": [1000000] * 30,
                }
            )
        ),
        "MSFT": Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [250.0] * 30,
                    "high": [251.0] * 30,
                    "low": [249.0] * 30,
                    "close": [250.5] * 30,
                    "volume": [800000] * 30,
                }
            )
        ),
        "GOOG": Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [100.0] * 30,
                    "high": [101.0] * 30,
                    "low": [99.0] * 30,
                    "close": [100.5] * 30,
                    "volume": [600000] * 30,
                }
            )
        ),
    }

    # Create earnings events
    events = [
        Event(
            timestamp=datetime(2023, 1, 5, 16, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="AAPL",
            payload={"eps": 1.5, "revenue": 1000000000},
        ),
        Event(
            timestamp=datetime(2023, 1, 10, 16, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="MSFT",
            payload={"eps": 2.0, "revenue": 2000000000},
        ),
    ]
    event_store = EventStore(events)

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        EarningsPlayStrategy,
        data=data,
        capital=100000.0,
        events=event_store,
    )

    print(f"Final equity: ${result.equity_series.iloc[-1]:,.2f}")
    print(f"Total return: {result.total_return:.2%}")


if __name__ == "__main__":
    main()

