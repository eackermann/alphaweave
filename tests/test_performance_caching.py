"""Tests for performance caching features."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class CachingTestStrategy(Strategy):
    """Strategy that uses indicators multiple times."""

    def init(self):
        self.call_count = 0

    def next(self, i):
        # Call sma multiple times - should hit cache
        sma1 = self.sma(period=20)
        sma2 = self.sma(period=20)  # Should use cache
        sma3 = self.sma(period=10)  # Different period, new computation

        # Verify results are consistent
        assert sma1 == sma2
        self.call_count += 1


def test_indicator_caching():
    """Test that indicators are cached and reused."""
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

    backtester = VectorBacktester()
    result = backtester.run(
        CachingTestStrategy,
        data=frame,
        capital=10000.0,
    )

    # Strategy should have run successfully
    assert result is not None


def test_clear_indicator_cache():
    """Test that clear_indicator_cache works."""
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

    strategy = CachingTestStrategy(frame)
    strategy._set_current_index(5)

    # Call indicator to populate cache
    _ = strategy.sma(period=20)
    assert len(strategy._indicator_cache) > 0

    # Clear cache
    strategy.clear_indicator_cache()
    assert len(strategy._indicator_cache) == 0


def test_resample_caching():
    """Test that resampled frames are cached."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 100,
        "high": [101.0] * 100,
        "low": [99.0] * 100,
        "close": [100.0] * 100,
        "volume": [1000] * 100,
    })
    frame = Frame.from_pandas(df)

    strategy = CachingTestStrategy(frame)

    # Get resampled frame twice - should use cache
    resampled1 = strategy.get_resampled_frame("_default", "1D")
    resampled2 = strategy.get_resampled_frame("_default", "1D")

    # Should be the same object (cached)
    assert resampled1 is resampled2

