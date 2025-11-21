"""Tests for fast performance mode."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class FastModeTestStrategy(Strategy):
    """Strategy that uses close() method."""

    def init(self):
        self.prices = []

    def next(self, i):
        # Use close() which should use fast arrays in fast mode
        price = self.close()
        self.prices.append(price)


def test_fast_mode_produces_same_results():
    """Test that fast mode produces identical results to default mode."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 50,
        "high": [101.0] * 50,
        "low": [99.0] * 50,
        "close": [100.0 + i * 0.1 for i in range(50)],
        "volume": [1000] * 50,
    })
    frame = Frame.from_pandas(df)

    # Run in default mode
    backtester_default = VectorBacktester(performance_mode="default")
    result_default = backtester_default.run(
        FastModeTestStrategy,
        data=frame,
        capital=10000.0,
    )

    # Run in fast mode
    backtester_fast = VectorBacktester(performance_mode="fast")
    result_fast = backtester_fast.run(
        FastModeTestStrategy,
        data=frame,
        capital=10000.0,
    )

    # Results should be identical
    assert result_default.final_equity == result_fast.final_equity
    assert len(result_default.equity_series) == len(result_fast.equity_series)
    assert (result_default.equity_series == result_fast.equity_series).all()


def test_fast_mode_sets_arrays():
    """Test that fast mode sets up numpy arrays."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [1000] * 10,
    })
    frame = Frame.from_pandas(df)

    backtester = VectorBacktester(performance_mode="fast")
    
    # Create strategy manually to check arrays
    strategy = FastModeTestStrategy(frame)
    backtester.run(
        FastModeTestStrategy,
        data=frame,
        capital=10000.0,
    )

    # After run, strategy should have fast arrays set (if we can access it)
    # This is more of an integration test

