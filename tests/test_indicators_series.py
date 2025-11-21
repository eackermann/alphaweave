"""Tests for indicators with Series, numpy array, and list sources."""

import pandas as pd
import numpy as np
from alphaweave.indicators.sma import SMA
from alphaweave.indicators.ema import EMA
from alphaweave.indicators.rsi import RSI
from alphaweave.indicators.roc import ROC


def test_sma_with_pandas_series():
    """Test SMA with pandas Series source."""
    series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    sma = SMA(series, period=3)
    
    # Should work without column parameter
    assert sma[0] == 100.0
    assert sma[2] == 101.0  # Average of [100, 101, 102]
    assert sma[9] == 108.0  # Average of [107, 108, 109] = 108


def test_sma_with_numpy_array():
    """Test SMA with numpy array source."""
    arr = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    sma = SMA(arr, period=3)
    
    assert sma[0] == 100.0
    assert sma[2] == 101.0
    assert sma[9] == 108.0  # Average of [107, 108, 109] = 108


def test_sma_with_list():
    """Test SMA with list source."""
    data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    sma = SMA(data, period=3)
    
    assert sma[0] == 100.0
    assert sma[2] == 101.0
    assert sma[9] == 108.0  # Average of [107, 108, 109] = 108


def test_ema_with_series():
    """Test EMA with pandas Series source."""
    series = pd.Series([100, 102, 104, 106, 108, 110])
    ema = EMA(series, period=3)
    
    assert not np.isnan(ema[0])
    assert ema[5] > ema[0]  # EMA should increase with increasing values


def test_rsi_with_series():
    """Test RSI with pandas Series source."""
    series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    rsi = RSI(series, period=5)
    
    values = rsi.values()
    assert len(values) == 10
    # RSI should be between 0 and 100
    assert all(0 <= v <= 100 for v in values if not np.isnan(v))


def test_roc_with_series():
    """Test ROC with pandas Series source."""
    series = pd.Series([100, 101, 102, 103, 104, 105])
    roc = ROC(series, period=1)
    
    # First value should be NaN (no previous value)
    assert np.isnan(roc[0])
    # Second value should be 1% (101/100 - 1)
    assert abs(roc[1] - 1.0) < 0.01


def test_series_with_index():
    """Test that Series index is preserved."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    series = pd.Series([100 + i for i in range(10)], index=dates)
    sma = SMA(series, period=3)
    
    values = sma.values()
    # Index should be preserved
    assert len(values) == 10
    assert values.index.equals(series.index)


def test_series_vs_frame_equivalence():
    """Test that Series and Frame (with column) give same results."""
    from alphaweave.core.frame import Frame
    
    # Create Frame with required columns
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
        "open": [99 + i for i in range(10)],
        "high": [101 + i for i in range(10)],
        "low": [98 + i for i in range(10)],
        "close": [100 + i for i in range(10)],
        "volume": [1000000] * 10,
    })
    frame = Frame.from_pandas(df)
    
    # Create Series from same data
    series = df["close"]
    
    # Both should give same SMA values
    sma_frame = SMA(frame, period=3, column="close")
    sma_series = SMA(series, period=3)
    
    # Compare values
    for i in range(10):
        assert abs(sma_frame[i] - sma_series[i]) < 1e-10

