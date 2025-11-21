"""Tests for backtester."""

import pandas as pd
from alphaweave.core.frame import Frame
from alphaweave.strategy.base import Strategy
from alphaweave.engine.vector import VectorBacktester


class BuyAndHold(Strategy):
    """Simple buy-and-hold strategy for testing."""

    def init(self):
        """Initialize strategy."""
        pass

    def next(self, i):
        """Buy and hold on first bar."""
        # Single asset called "ASSET"
        self.order_target_percent("ASSET", 1.0)


def make_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "datetime": pd.date_range("2021-01-01", periods=10, freq="D"),
        "open": [10 + i for i in range(10)],
        "high": [11 + i for i in range(10)],
        "low": [9 + i for i in range(10)],
        "close": [10 + i for i in range(10)],
        "volume": [100] * 10,
    })


def test_vector_backtester_runs_simple_strategy():
    """Test that VectorBacktester runs a simple strategy."""
    df = make_data()
    frame = Frame.from_pandas(df)
    res = VectorBacktester().run(BuyAndHold, data={"ASSET": frame}, capital=1000.0)
    assert hasattr(res, "equity_series")
    # equity_series can be list or Series
    assert len(res.equity_series) == len(df)
    assert len(res.trades) >= 1
    # Equity should be >= 0
    assert all(e >= 0 for e in res.equity_series)


def test_vector_backtester_equity_tracking():
    """Test that equity tracking works correctly."""
    df = make_data()
    frame = Frame.from_pandas(df)
    res = VectorBacktester().run(BuyAndHold, data={"ASSET": frame}, capital=1000.0)
    
    # First equity should be starting capital (before any trades)
    assert res.equity_series[0] == 1000.0
    
    # Equity should change as price changes
    # Since we buy at bar 0 close price (10) and hold, equity should track price
    # After first trade, equity = cash + position_value
    # We should have some trades
    assert len(res.trades) > 0

