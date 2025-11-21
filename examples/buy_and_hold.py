"""Buy-and-hold strategy example."""

import alphaweave as aw
import pandas as pd

# Create sample data
df = pd.DataFrame({
    "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
    "open": [10 + i for i in range(10)],
    "high": [11 + i for i in range(10)],
    "low": [9 + i for i in range(10)],
    "close": [10 + i for i in range(10)],
    "volume": [100] * 10,
})

frame = aw.core.frame.Frame.from_pandas(df)


class BuyAndHold(aw.strategy.base.Strategy):
    """Simple buy-and-hold strategy."""

    def init(self):
        """Initialize strategy."""
        pass

    def next(self, i):
        """Buy and hold on first bar."""
        self.order_target_percent("TEST", 1.0)


if __name__ == "__main__":
    res = aw.engine.vector.VectorBacktester().run(
        BuyAndHold, data={"TEST": frame}, capital=1000
    )
    print(f"Equity series length: {len(res.equity_series)}")
    print(f"Number of trades: {len(res.trades)}")
    print(f"Final equity: {res.equity_series[-1]:.2f}")
    print(f"Equity series: {res.equity_series}")

