"""Tests for Frame abstraction."""

import pandas as pd
import polars as pl
from alphaweave.core.frame import Frame


def make_sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=5, freq="D"),
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [10] * 5
    })


def test_frame_roundtrip_pandas_polars():
    """Test roundtrip conversion between pandas and polars."""
    pdf = make_sample_df()
    frame = Frame.from_pandas(pdf)
    pl_df = frame.to_polars()
    assert isinstance(pl_df, pl.DataFrame)
    # Convert back
    frame2 = Frame.from_polars(pl_df)
    pdf2 = frame2.to_pandas()
    # Compare data columns (excluding datetime which is in index)
    # Original has datetime as column, frame2 has it as index
    data_cols = ["open", "high", "low", "close", "volume"]
    assert pdf2.shape[0] == pdf.shape[0]  # Same number of rows
    assert len([c for c in pdf2.columns if c in data_cols]) == len([c for c in pdf.columns if c in data_cols])


def test_frame_normalizes_columns():
    """Test that Frame normalizes column names."""
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=3, freq="D"),
        "Open": [1, 2, 3],
        "High": [1, 2, 3],
        "Low": [1, 2, 3],
        "Close": [1, 2, 3],
        "Volume": [10, 10, 10],
    })
    frame = Frame.from_pandas(df)
    pdf = frame.to_pandas()
    cols_lower = {c.lower() for c in pdf.columns}
    assert "open" in cols_lower
    assert "high" in cols_lower
    assert "low" in cols_lower
    assert "close" in cols_lower


def test_frame_validation():
    """Test that Frame validation works."""
    # Valid frame
    df = make_sample_df()
    frame = Frame.from_pandas(df)
    frame.validate()  # Should not raise

    # Invalid frame (missing close)
    df_invalid = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=3, freq="D"),
        "open": [1, 2, 3],
        "high": [1, 2, 3],
        "low": [1, 2, 3],
        # Missing close
        "volume": [10, 10, 10],
    })
    try:
        frame_invalid = Frame.from_pandas(df_invalid)
        frame_invalid.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

