"""Tests for Polars compatibility."""

import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

import pytest

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
def test_polars_frame_backtest():
    """Test that backtest works with Polars-backed Frame."""
    # Create Polars DataFrame
    dates = pl.date_range(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-20"),
        interval="1d",
        eager=True,
    )
    pl_df = pl.DataFrame({
        "datetime": dates,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0 + i * 0.1 for i in range(20)],
        "volume": [1000] * 20,
    })

    # Create Frame from Polars
    frame = Frame.from_polars(pl_df)

    # Verify it's Polars-backed
    assert frame.backend == "polars"

    # Create simple strategy
    class SimpleStrategy(Strategy):
        def init(self):
            pass

        def next(self, i):
            if i == 0:
                self.order_target_percent("_default", 1.0)

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        SimpleStrategy,
        data=frame,
        capital=10000.0,
    )

    # Should complete successfully
    assert result is not None
    assert result.final_equity > 0


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
def test_polars_frame_strategy_helpers():
    """Test that Strategy helpers work with Polars-backed Frame."""
    dates = pl.date_range(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-20"),
        interval="1d",
        eager=True,
    )
    pl_df = pl.DataFrame({
        "datetime": dates,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [1000] * 20,
    })

    frame = Frame.from_polars(pl_df)

    class HelperTestStrategy(Strategy):
        def init(self):
            pass

        def next(self, i):
            # Test that close() works
            price = self.close()
            assert price == 100.0

    backtester = VectorBacktester()
    result = backtester.run(
        HelperTestStrategy,
        data=frame,
        capital=10000.0,
    )

    assert result is not None

