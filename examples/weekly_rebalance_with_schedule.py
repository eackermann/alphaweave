"""
Example: Weekly rebalance using schedule.every("1W") + at_open().

This strategy demonstrates how to use the Schedule helper to rebalance
a portfolio weekly at the market open.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.data.loaders import load_csv
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class WeeklyRebalanceStrategy(Strategy):
    """Rebalance portfolio weekly at market open."""

    def init(self):
        # Target allocation
        self.target_weights = {"SPY": 0.6, "QQQ": 0.4}

    def next(self, i):
        # Only rebalance once per week, at session open
        if not (self.schedule.every("1W") and self.schedule.at_open()):
            return

        # Rebalance to target weights
        for symbol, weight in self.target_weights.items():
            self.order_target_percent(symbol, weight)


def main():
    # Load data (example - adjust paths as needed)
    # data = {
    #     "SPY": load_csv("data/SPY_daily.csv", tz="America/New_York"),
    #     "QQQ": load_csv("data/QQQ_daily.csv", tz="America/New_York"),
    # }

    # For demonstration, create synthetic data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "SPY": Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [400.0] * 100,
                    "high": [401.0] * 100,
                    "low": [399.0] * 100,
                    "close": [400.5] * 100,
                    "volume": [1000000] * 100,
                }
            )
        ),
        "QQQ": Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [300.0] * 100,
                    "high": [301.0] * 100,
                    "low": [299.0] * 100,
                    "close": [300.5] * 100,
                    "volume": [500000] * 100,
                }
            )
        ),
    }

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        WeeklyRebalanceStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.equity_series.iloc[-1]:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()

