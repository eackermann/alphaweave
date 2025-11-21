"""
Download daily OHLCV data for a few tickers using yfinance
and save them as CSVs compatible with alphaweave.Frame.from_pandas.

Tickers:
- SPY  (S&P 500 ETF)
- ^NDX (Nasdaq 100 index)
- TQQQ (3x bull Nasdaq 100 ETF)
- SQQQ (3x bear Nasdaq 100 ETF)
"""

import os
from datetime import datetime
import pandas as pd
import yfinance as yf


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


TICKERS = {
    "SPY": "SPY",
    "NDX": "^NDX",   # Yahoo's symbol for Nasdaq 100 index
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
}


def download_ticker(symbol: str, yf_symbol: str, years: int = 5) -> None:
    """
    Download daily OHLCV for the last `years` years for yf_symbol,
    save as data/{symbol}_daily.csv with columns:
    datetime, open, high, low, close, volume
    """
    end = datetime.today()
    start = datetime(end.year - years, end.month, end.day)

    print(f"Downloading {symbol} ({yf_symbol}) from {start.date()} to {end.date()}...")
    df = yf.download(yf_symbol, start=start, end=end, interval="1d", auto_adjust=False)

    if df.empty:
        print(f"WARNING: no data returned for {symbol}")
        return

    # yfinance returns index as DatetimeIndex, columns: Open, High, Low, Close, Adj Close, Volume
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.insert(0, "datetime", df.index)  # add datetime column

    out_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {symbol} to {out_path}, rows: {len(df)}")


def main():
    for symbol, yf_symbol in TICKERS.items():
        download_ticker(symbol, yf_symbol, years=5)


if __name__ == "__main__":
    main()
