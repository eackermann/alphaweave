"""Tests for split adjustment detection and filtering."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.data.corporate_actions import (
    SplitAction,
    detect_split_adjustment_for_symbol,
    auto_filter_splits_for_data,
    build_corporate_actions_store_auto,
    build_corporate_actions_store,
)
from datetime import datetime


def test_detect_split_adjustment_raw_data():
    """Test detection on raw (unadjusted) price data."""
    # Create data with a visible 2-for-1 split
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    
    # Pre-split: price 100
    # Post-split: price 50 (raw, not adjusted)
    prices = []
    for i, date in enumerate(dates):
        if date < pd.Timestamp("2021-01-11"):
            prices.append(100.0 + i * 0.1)
        else:
            # After split on 2021-01-11, price drops to ~50
            prices.append(50.0 + (i - 10) * 0.05)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    
    # Create a 2-for-1 split on 2021-01-11
    split = SplitAction(symbol="TEST", date=datetime(2021, 1, 11), ratio=2.0)
    
    is_adjusted, diag = detect_split_adjustment_for_symbol(
        frame, [split], price_column="close"
    )
    
    # Should detect as raw (not adjusted) because ratio_est should be close to 2.0
    assert is_adjusted is False
    assert len(diag) == 1
    assert diag.iloc[0]["ratio"] == 2.0
    # ratio_est should be close to 2.0 (p_before / p_after ≈ 100 / 50 = 2.0)
    assert abs(diag.iloc[0]["ratio_est"] - 2.0) < 0.5


def test_detect_split_adjustment_already_adjusted():
    """Test detection on already-adjusted price data."""
    # Create data that's already split-adjusted (no discontinuity)
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    
    # Prices continue smoothly (already adjusted)
    prices = [100.0 + i * 0.1 for i in range(20)]
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    
    # Create a 2-for-1 split on 2021-01-11
    split = SplitAction(symbol="TEST", date=datetime(2021, 1, 11), ratio=2.0)
    
    is_adjusted, diag = detect_split_adjustment_for_symbol(
        frame, [split], price_column="close"
    )
    
    # Should detect as already adjusted because ratio_est should be close to 1.0
    assert is_adjusted is True
    assert len(diag) == 1
    # ratio_est should be close to 1.0 (p_before / p_after ≈ 1.0)
    assert abs(diag.iloc[0]["ratio_est"] - 1.0) < 0.5


def test_auto_filter_splits_for_data():
    """Test automatic filtering of splits."""
    # Create two frames: one raw, one adjusted
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    
    # Raw data with visible split
    raw_prices = []
    for i, date in enumerate(dates):
        if date < pd.Timestamp("2021-01-11"):
            raw_prices.append(100.0 + i * 0.1)
        else:
            raw_prices.append(50.0 + (i - 10) * 0.05)
    
    raw_df = pd.DataFrame({
        "datetime": dates,
        "open": raw_prices,
        "high": [p * 1.01 for p in raw_prices],
        "low": [p * 0.99 for p in raw_prices],
        "close": raw_prices,
        "volume": [1000] * 20,
    })
    raw_frame = Frame.from_pandas(raw_df)
    
    # Adjusted data (smooth)
    adj_prices = [100.0 + i * 0.1 for i in range(20)]
    adj_df = pd.DataFrame({
        "datetime": dates,
        "open": adj_prices,
        "high": [p * 1.01 for p in adj_prices],
        "low": [p * 0.99 for p in adj_prices],
        "close": adj_prices,
        "volume": [1000] * 20,
    })
    adj_frame = Frame.from_pandas(adj_df)
    
    data = {
        "RAW": raw_frame,
        "ADJ": adj_frame,
    }
    
    splits_by_symbol = {
        "RAW": [SplitAction(symbol="RAW", date=datetime(2021, 1, 11), ratio=2.0)],
        "ADJ": [SplitAction(symbol="ADJ", date=datetime(2021, 1, 11), ratio=2.0)],
    }
    
    filtered = auto_filter_splits_for_data(data, splits_by_symbol)
    
    # RAW should keep its split (not adjusted)
    assert len(filtered["RAW"]) == 1
    
    # ADJ should have its split filtered out (already adjusted)
    assert len(filtered["ADJ"]) == 0


def test_build_corporate_actions_store_auto():
    """Test convenience function for building store with auto-filtering."""
    # Create adjusted data
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    prices = [100.0 + i * 0.1 for i in range(20)]
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    data = {"TEST": frame}
    
    splits = [SplitAction(symbol="TEST", date=datetime(2021, 1, 11), ratio=2.0)]
    dividends = []  # No dividends for this test
    
    store = build_corporate_actions_store_auto(
        data=data,
        splits=splits,
        dividends=dividends,
    )
    
    # Split should be filtered out (data is already adjusted)
    splits_on_date = store.get_splits_on_date("TEST", datetime(2021, 1, 11))
    assert len(splits_on_date) == 0


def test_build_corporate_actions_store_auto_preserves_raw_splits():
    """Test that raw (unadjusted) splits are preserved."""
    # Create raw data with visible split
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    prices = []
    for i, date in enumerate(dates):
        if date < pd.Timestamp("2021-01-11"):
            prices.append(100.0 + i * 0.1)
        else:
            prices.append(50.0 + (i - 10) * 0.05)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    data = {"TEST": frame}
    
    splits = [SplitAction(symbol="TEST", date=datetime(2021, 1, 11), ratio=2.0)]
    dividends = []
    
    store = build_corporate_actions_store_auto(
        data=data,
        splits=splits,
        dividends=dividends,
    )
    
    # Split should be preserved (data is raw)
    splits_on_date = store.get_splits_on_date("TEST", datetime(2021, 1, 11))
    assert len(splits_on_date) == 1
    assert splits_on_date[0].ratio == 2.0

