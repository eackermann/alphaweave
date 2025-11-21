"""Tests for Strategy helper methods."""

import pandas as pd
from alphaweave.core.frame import Frame
from alphaweave.strategy.base import Strategy
from alphaweave.indicators.sma import SMA


def make_sample_data():
    """Create sample data for testing."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
        "open": [100 + i for i in range(10)],
        "high": [102 + i for i in range(10)],
        "low": [98 + i for i in range(10)],
        "close": [101 + i for i in range(10)],
        "volume": [1000000] * 10,
    })
    return Frame.from_pandas(df)


class HelperTestStrategy(Strategy):
    """Test strategy that uses helper methods."""

    def next(self, index):
        """Store values for testing."""
        self.test_close = self.close()
        self.test_sma = self.sma(period=3)


def test_strategy_close():
    """Test that Strategy.close() returns the current bar's close."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    # Set current index and call next
    strategy._set_current_index(0)
    strategy.next(0)
    assert strategy.test_close == 101.0  # First close value
    
    strategy._set_current_index(5)
    strategy.next(5)
    assert strategy.test_close == 106.0  # 6th close value (index 5)


def test_strategy_close_with_symbol():
    """Test Strategy.close() with symbol parameter."""
    frame = make_sample_data()
    strategy = HelperTestStrategy({"_default": frame})
    strategy.init()
    
    strategy._set_current_index(3)
    strategy.next(3)
    assert strategy.test_close == 104.0


def test_strategy_sma():
    """Test that Strategy.sma() returns the same value as direct indicator indexing."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    # Get the close series
    close_series = strategy.series(field="close")
    
    # Create SMA indicator directly
    sma_indicator = SMA(close_series, period=3)
    
    # Test at index 5
    strategy._set_current_index(5)
    strategy.next(5)
    
    # Strategy.sma() should return the same value as direct indicator access
    strategy_sma = strategy.test_sma
    direct_sma = sma_indicator[5]
    
    assert abs(strategy_sma - direct_sma) < 1e-10, \
        f"Strategy.sma()={strategy_sma} != direct SMA[5]={direct_sma}"


def test_strategy_sma_caching():
    """Test that indicators are cached and reused."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    # Call sma() multiple times - should use cached indicator
    strategy._set_current_index(5)
    sma1 = strategy.sma(period=3)
    sma2 = strategy.sma(period=3)  # Should use cache
    
    assert abs(sma1 - sma2) < 1e-10
    
    # Cache should have one entry
    assert len(strategy._indicator_cache) == 1
    
    # Different period should create new cache entry
    sma3 = strategy.sma(period=5)
    assert len(strategy._indicator_cache) == 2


def test_strategy_series():
    """Test Strategy.series() method."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    close_series = strategy.series(field="close")
    assert isinstance(close_series, pd.Series)
    assert len(close_series) == 10
    assert close_series.iloc[0] == 101.0
    
    open_series = strategy.series(field="open")
    assert len(open_series) == 10
    assert open_series.iloc[0] == 100.0


def test_strategy_ema():
    """Test Strategy.ema() method."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    strategy._set_current_index(5)
    ema_value = strategy.ema(period=3)
    
    # EMA should be a valid number
    assert not pd.isna(ema_value)
    assert ema_value > 0


def test_strategy_helpers_without_index():
    """Test that helpers raise error if _current_index is not set."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    # Should raise error if _current_index is not set
    try:
        strategy.close()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "_current_index" in str(e)
    
    try:
        strategy.sma(period=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "_current_index" in str(e)


def test_strategy_helpers_different_fields():
    """Test that helpers work with different fields."""
    frame = make_sample_data()
    strategy = HelperTestStrategy(frame)
    strategy.init()
    
    strategy._set_current_index(5)
    
    # Test with different fields
    close_sma = strategy.sma(period=3, field="close")
    open_sma = strategy.sma(period=3, field="open")
    
    # Should be different (close and open have different values)
    assert abs(close_sma - open_sma) > 0.1


def test_strategy_helpers_multi_symbol():
    """Test helpers with multiple symbols."""
    frame1 = make_sample_data()
    frame2 = make_sample_data()
    
    strategy = HelperTestStrategy({"AAPL": frame1, "MSFT": frame2})
    strategy.init()
    
    strategy._set_current_index(5)
    
    # Should work with symbol parameter
    aapl_close = strategy.close("AAPL")
    msft_close = strategy.close("MSFT")
    
    assert aapl_close == 106.0
    assert msft_close == 106.0

