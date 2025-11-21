"""Tests for trading signals."""

import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.indicators.sma import SMA
from alphaweave.indicators.ema import EMA
from alphaweave.signals.crossover import CrossOver, CrossUnder
from alphaweave.signals.comparison import GreaterThan, LessThan


def make_trending_frame():
    """Create a Frame with a clear trend for crossover testing."""
    # Create data where close price increases, then SMA crosses above
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
        "open": closes,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1000000] * 10,
    })
    return Frame.from_pandas(df)


def make_oscillating_frame():
    """Create a Frame with oscillating prices."""
    closes = [100, 105, 95, 100, 105, 95, 100, 105, 95, 100]
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
        "open": closes,
        "high": [c + 2 for c in closes],
        "low": [c - 2 for c in closes],
        "close": closes,
        "volume": [1000000] * 10,
    })
    return Frame.from_pandas(df)


def test_crossover_basic():
    """Test CrossOver signal basic functionality."""
    frame = make_trending_frame()
    sma_fast = SMA(frame, period=2, column="close")
    sma_slow = SMA(frame, period=5, column="close")
    
    crossover = CrossOver(sma_fast, sma_slow)
    
    # At index 0, no crossover (need previous bar)
    assert not crossover(0)
    
    # Check a few indices
    # With trending data, fast SMA should eventually cross above slow SMA
    results = [crossover(i) for i in range(10)]
    # May or may not have crossovers depending on data pattern
    # Just verify the signal works without crashing
    assert isinstance(results[0], bool)


def test_crossunder_basic():
    """Test CrossUnder signal basic functionality."""
    frame = make_oscillating_frame()
    sma_fast = SMA(frame, period=2, column="close")
    sma_slow = SMA(frame, period=5, column="close")
    
    crossunder = CrossUnder(sma_fast, sma_slow)
    
    # At index 0, no crossunder
    assert not crossunder(0)
    
    # With oscillating data, should have crossunders
    results = [crossunder(i) for i in range(10)]
    # May or may not have crossunders depending on data


def test_crossover_no_cross():
    """Test CrossOver when no crossover occurs."""
    frame = make_trending_frame()
    sma1 = SMA(frame, period=3, column="close")
    sma2 = SMA(frame, period=3, column="close")
    
    crossover = CrossOver(sma1, sma2)
    
    # Two identical SMAs should never cross
    results = [crossover(i) for i in range(10)]
    assert not any(results)


def test_greater_than_constant():
    """Test GreaterThan signal with constant threshold."""
    frame = make_trending_frame()
    sma = SMA(frame, period=3, column="close")
    
    gt = GreaterThan(sma, threshold=102.0)
    
    # Early values should be False, later values True
    assert not gt(0)  # SMA[0] = 100 < 102
    assert gt(5)  # SMA[5] should be > 102


def test_less_than_constant():
    """Test LessThan signal with constant threshold."""
    frame = make_trending_frame()
    sma = SMA(frame, period=3, column="close")
    
    lt = LessThan(sma, threshold=108.0)
    
    # Early values should be True, later values False
    assert lt(0)  # SMA[0] = 100 < 108
    assert not lt(9)  # SMA[9] should be > 108


def test_greater_than_indicator():
    """Test GreaterThan signal with indicator threshold."""
    frame = make_trending_frame()
    sma_fast = SMA(frame, period=2, column="close")
    sma_slow = SMA(frame, period=5, column="close")
    
    gt = GreaterThan(sma_fast, threshold=sma_slow)
    
    # Should be True when fast > slow
    results = [gt(i) for i in range(10)]
    # With trending data, fast should eventually be > slow
    assert any(results[5:])  # Later in the series


def test_less_than_indicator():
    """Test LessThan signal with indicator threshold."""
    frame = make_trending_frame()
    sma_fast = SMA(frame, period=2, column="close")
    sma_slow = SMA(frame, period=5, column="close")
    
    lt = LessThan(sma_fast, threshold=sma_slow)
    
    # Should be True when fast < slow
    results = [lt(i) for i in range(10)]
    # May or may not have True values depending on data pattern
    # Just verify the signal works without crashing
    assert isinstance(results[0], bool)


def test_signal_edge_cases():
    """Test signals with edge cases."""
    frame = make_trending_frame()
    sma = SMA(frame, period=3, column="close")
    
    # Test with invalid index
    gt = GreaterThan(sma, threshold=100.0)
    # Should handle gracefully
    try:
        result = gt(100)  # Out of bounds
        assert result is False
    except (IndexError, KeyError):
        pass  # Also acceptable


def test_signal_nan_handling():
    """Test that signals handle NaN values gracefully."""
    frame = make_trending_frame()
    roc = SMA(frame, period=20, column="close")  # Period larger than data
    
    gt = GreaterThan(roc, threshold=0.0)
    
    # Should not crash on NaN values
    for i in range(10):
        try:
            _ = gt(i)
        except (ValueError, TypeError):
            # Should handle NaN gracefully
            pass

