"""Tests for technical indicators."""

import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.indicators.sma import SMA
from alphaweave.indicators.ema import EMA
from alphaweave.indicators.rsi import RSI
from alphaweave.indicators.roc import ROC
from alphaweave.indicators.atr import ATR


def make_sample_frame():
    """Create a sample Frame for testing."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=20, freq="D"),
        "open": [100 + i - 0.5 for i in range(20)],  # Open slightly lower
        "high": [102 + i for i in range(20)],
        "low": [98 + i for i in range(20)],
        "close": [100 + i for i in range(20)],
        "volume": [1000000] * 20,
    })
    return Frame.from_pandas(df)


def test_sma_basic():
    """Test SMA indicator basic functionality."""
    frame = make_sample_frame()
    sma = SMA(frame, period=5, column="close")
    
    # Test lazy evaluation
    val0 = sma[0]
    assert not np.isnan(val0)
    assert val0 == 100.0  # First value should be the first close
    
    # Test all values
    values = sma.values()
    assert len(values) == 20
    assert values.iloc[0] == 100.0
    assert values.iloc[4] == 102.0  # Average of first 5 values


def test_sma_period():
    """Test SMA with different periods."""
    frame = make_sample_frame()
    sma10 = SMA(frame, period=10, column="close")
    
    values = sma10.values()
    assert len(values) == 20
    # First 9 values should be partial averages
    assert values.iloc[9] == 104.5  # Average of values 0-9 (100 to 109)


def test_ema_basic():
    """Test EMA indicator basic functionality."""
    frame = make_sample_frame()
    ema = EMA(frame, period=5, column="close")
    
    val0 = ema[0]
    assert not np.isnan(val0)
    
    values = ema.values()
    assert len(values) == 20
    # EMA should give more weight to recent values
    assert values.iloc[0] == 100.0


def test_rsi_basic():
    """Test RSI indicator basic functionality."""
    frame = make_sample_frame()
    rsi = RSI(frame, period=14, column="close")
    
    # RSI should be between 0 and 100
    values = rsi.values()
    assert len(values) == 20
    assert all(0 <= v <= 100 for v in values if not np.isnan(v))
    
    # RSI should be valid (not NaN for most values)
    valid_values = [v for v in values if not np.isnan(v)]
    assert len(valid_values) > 0


def test_rsi_period():
    """Test RSI with different periods."""
    frame = make_sample_frame()
    rsi = RSI(frame, period=5, column="close")
    
    values = rsi.values()
    assert len(values) == 20
    assert all(0 <= v <= 100 for v in values if not np.isnan(v))


def test_roc_basic():
    """Test ROC indicator basic functionality."""
    frame = make_sample_frame()
    roc = ROC(frame, period=1, column="close")
    
    # ROC with period 1 should be 0 for first bar, then percentage change
    val0 = roc[0]
    assert np.isnan(val0) or val0 == 0.0
    
    val1 = roc[1]
    # Close goes from 100 to 101, so ROC should be 1%
    assert abs(val1 - 1.0) < 0.01


def test_roc_period():
    """Test ROC with different periods."""
    frame = make_sample_frame()
    roc = ROC(frame, period=5, column="close")
    
    values = roc.values()
    assert len(values) == 20
    # First 5 values should be NaN
    assert np.isnan(values.iloc[4])
    # 6th value should have ROC from 0 to 5
    assert not np.isnan(values.iloc[5])


def test_atr_basic():
    """Test ATR indicator basic functionality."""
    frame = make_sample_frame()
    atr = ATR(frame, period=14)
    
    values = atr.values()
    assert len(values) == 20
    # ATR should always be positive
    assert all(v >= 0 for v in values if not np.isnan(v))
    
    # With constant high-low spread of 4, ATR should be around 4
    assert abs(atr[19] - 4.0) < 1.0


def test_atr_period():
    """Test ATR with different periods."""
    frame = make_sample_frame()
    atr = ATR(frame, period=5)
    
    values = atr.values()
    assert len(values) == 20
    assert all(v >= 0 for v in values if not np.isnan(v))


def test_indicator_lazy_evaluation():
    """Test that indicators use lazy evaluation."""
    frame = make_sample_frame()
    sma = SMA(frame, period=5)
    
    # Accessing a single value should compute all values
    _ = sma[10]
    
    # Now accessing another value should be fast (already computed)
    val = sma[15]
    assert not np.isnan(val)


def test_indicator_different_columns():
    """Test indicators with different column names."""
    frame = make_sample_frame()
    sma_close = SMA(frame, period=5, column="close")
    sma_open = SMA(frame, period=5, column="open")
    
    # They should have different values
    assert sma_close[10] != sma_open[10]


def test_indicator_invalid_period():
    """Test that invalid periods raise errors."""
    frame = make_sample_frame()
    
    try:
        SMA(frame, period=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        RSI(frame, period=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

