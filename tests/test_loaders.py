"""Tests for data loaders."""

import pandas as pd

from alphaweave.data.loaders import load_csv, load_parquet, load_directory
from alphaweave.core.frame import Frame


def _make_dummy_ohlcv(n=5):
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
    })


def test_load_csv_with_symbol_header_row(tmp_path):
    # Create a CSV with a second row of symbol names (vendor-style)
    df = _make_dummy_ohlcv(3)

    # Write CSV with standard header
    csv_path = tmp_path / "SPY_daily.csv"
    df.to_csv(csv_path, index=False)

    # Now manually insert a symbol row after the header:
    # datetime,open,high,low,close,volume
    # ,SPY,SPY,SPY,SPY,SPY
    text = csv_path.read_text().splitlines()
    header = text[0]
    body = text[1:]
    symbol_row = ",SPY,SPY,SPY,SPY,SPY"
    new_text = "\n".join([header, symbol_row] + body)
    csv_path.write_text(new_text)

    frame = load_csv(str(csv_path), symbol=None)
    assert isinstance(frame, Frame)

    pdf = frame.to_pandas()
    # Should have dropped the symbol row and kept 3 data rows
    assert len(pdf) == 3
    # Symbol should be inferred from filename and attached
    assert "symbol" in pdf.columns
    assert pdf["symbol"].iloc[0] == "SPY"


def test_load_parquet_basic(tmp_path):
    df = _make_dummy_ohlcv(4)
    pq_path = tmp_path / "tqqq.parquet"
    df.to_parquet(pq_path)

    frame = load_parquet(str(pq_path), symbol=None)
    assert isinstance(frame, Frame)

    pdf = frame.to_pandas()
    assert len(pdf) == 4
    assert "open" in pdf.columns
    assert "close" in pdf.columns
    # Symbol inferred from filename: "tqqq" -> "TQQQ"
    assert "symbol" in pdf.columns
    assert pdf["symbol"].iloc[0] == "TQQQ"


def test_load_directory_loads_multiple_symbols(tmp_path):
    # Create two CSVs with simple OHLCV data
    df_spy = _make_dummy_ohlcv(2)
    df_tqqq = _make_dummy_ohlcv(3)

    spy_path = tmp_path / "SPY_daily.csv"
    tqqq_path = tmp_path / "TQQQ_daily.csv"

    df_spy.to_csv(spy_path, index=False)
    df_tqqq.to_csv(tqqq_path, index=False)

    data = load_directory(tmp_path)

    # Should have keys for SPY and TQQQ
    assert set(data.keys()) == {"SPY", "TQQQ"}
    assert isinstance(data["SPY"], Frame)
    assert isinstance(data["TQQQ"], Frame)

    spy_pdf = data["SPY"].to_pandas()
    tqqq_pdf = data["TQQQ"].to_pandas()

    assert len(spy_pdf) == 2
    assert len(tqqq_pdf) == 3
    assert "open" in spy_pdf.columns
    assert "close" in tqqq_pdf.columns
    assert "symbol" in spy_pdf.columns
    assert spy_pdf["symbol"].iloc[0] == "SPY"
    assert tqqq_pdf["symbol"].iloc[0] == "TQQQ"


def test_load_directory_with_symbol_filter(tmp_path):
    df_spy = _make_dummy_ohlcv(2)
    df_tqqq = _make_dummy_ohlcv(2)
    df_sqqq = _make_dummy_ohlcv(2)

    (tmp_path / "SPY_daily.csv").write_text(df_spy.to_csv(index=False))
    (tmp_path / "TQQQ_daily.csv").write_text(df_tqqq.to_csv(index=False))
    (tmp_path / "SQQQ_daily.csv").write_text(df_sqqq.to_csv(index=False))

    data = load_directory(tmp_path, symbols=["SPY", "SQQQ"])

    assert set(data.keys()) == {"SPY", "SQQQ"}
    assert "TQQQ" not in data
