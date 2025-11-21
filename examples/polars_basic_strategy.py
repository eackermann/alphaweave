"""
Example: Basic strategy using Polars-backed Frame.

This demonstrates that alphaweave works seamlessly with Polars DataFrames
via the Frame abstraction.
"""

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Polars not available. Install with: pip install polars")

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class SimplePolarsStrategy(Strategy):
    """Simple strategy that works with Polars-backed data."""

    def init(self):
        pass

    def next(self, i):
        # Strategy logic works the same regardless of backend
        if i == 0:
            # Buy at start
            self.order_target_percent("_default", 1.0)


def main():
    if not POLARS_AVAILABLE:
        print("Polars not available. Skipping example.")
        return

    # Create data using Polars
    dates = pl.date_range(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-12-31"),
        interval="1d",
        eager=True,
    )

    # Generate synthetic price data
    import numpy as np
    np.random.seed(42)
    n = len(dates)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    pl_df = pl.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": [1000000] * n,
    })

    # Create Frame from Polars DataFrame
    frame = Frame.from_polars(pl_df)

    print(f"Frame backend: {frame.backend}")
    print(f"Data shape: {pl_df.shape}")

    # Run backtest - works seamlessly!
    backtester = VectorBacktester()
    result = backtester.run(
        SimplePolarsStrategy,
        data=frame,
        capital=100000.0,
    )

    print(f"\nBacktest Results:")
    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")

    # Strategy helpers work too
    strategy = SimplePolarsStrategy(frame)
    strategy._set_current_index(0)
    strategy._set_current_timestamp(pd.Timestamp("2023-01-01"))
    price = strategy.close()
    print(f"\nPrice at first bar: ${price:.2f}")


if __name__ == "__main__":
    main()

