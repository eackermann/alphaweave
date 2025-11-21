"""Data loaders for CSV and Parquet files."""

import os
import re
from typing import Dict, Sequence, Optional
from pathlib import Path
import pandas as pd
from alphaweave.core.frame import Frame


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _infer_symbol_from_path(path: str) -> Optional[str]:
    """
    Infer a symbol name from the filename.

    Examples:
        data/SPY_daily.csv      -> SPY
        /foo/bar/ndx.parquet    -> NDX
        tqqq-1d.parquet         -> TQQQ
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    # Split on common separators and take the first token
    token = re.split(r"[_\-\.]", name)[0]
    token = token.strip()

    if not token:
        return None

    return token.upper()


def _normalize_ohlcv_dataframe(
    df: pd.DataFrame,
    path: str = "<in-memory>",
    prefer_adj_close: bool = True,
) -> pd.DataFrame:
    """
    Normalize a raw DataFrame into the format expected by Frame:

      - Ensure a datetime index
      - Normalize column names: datetime, open, high, low, close, adj_close, volume
      - Use adjusted close as 'close' if available (optionally keep raw close)
      - Drop obvious vendor header/garbage rows
      - Enforce float dtype for OHLCV

    Expected output:
        index: datetime (sorted)
        columns: at least ['open', 'high', 'low', 'close', 'volume']
                 optionally 'adj_close', 'close_raw', 'symbol'
    """

    # --- Normalize column names (case-insensitive) ---
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("datetime", "date", "time", "timestamp"):
            rename_map[c] = "datetime"
        elif lc in ("open", "o"):
            rename_map[c] = "open"
        elif lc in ("high", "h"):
            rename_map[c] = "high"
        elif lc in ("low", "l"):
            rename_map[c] = "low"
        elif lc in ("close", "c", "last", "closing"):
            rename_map[c] = "close"
        elif lc in ("adj close", "adj_close", "adjusted close", "adjusted_close", "close_adj"):
            rename_map[c] = "adj_close"
        elif lc in ("volume", "v", "vol", "volumne"):
            rename_map[c] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Ensure we have a datetime column ---
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "datetime"})
        else:
            raise ValueError(
                f"{path}: no 'datetime' column or datetime-like index found."
            )

    # Parse datetime and drop invalid
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # --- Drop obvious vendor header rows / garbage ---
    # e.g., rows where OHLCV columns are all NaN or all non-numeric
    candidate_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    if candidate_cols:
        # Try coerce to numeric temporarily to find junk rows
        temp = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
        # Drop rows where all OHLCV-like columns are NaN
        df = df[~temp.isna().all(axis=1)]

    # --- Ensure required columns exist ---
    # We allow either close or adj_close (we can derive close from adj_close)
    required_base = ["open", "high", "low", "volume"]
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}' after normalization.")

    if "close" not in df.columns and "adj_close" not in df.columns:
        raise ValueError(f"{path}: missing both 'close' and 'adj_close' columns.")

    # --- Adjusted close handling ---
    # If we have adj_close and prefer_adj_close, we:
    #   - store existing 'close' (if any) as 'close_raw'
    #   - overwrite 'close' with 'adj_close'
    if "adj_close" in df.columns and prefer_adj_close:
        if "close" in df.columns:
            df["close_raw"] = pd.to_numeric(df["close"], errors="coerce")
        df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    else:
        # No adj_close or not preferring it: rely on 'close' as-is
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # --- Enforce numeric OHLCV types and drop bad rows ---
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Index by datetime
    df = df.set_index("datetime").sort_index()

    return df


def _attach_symbol(df: pd.DataFrame, symbol: Optional[str], path: str) -> pd.DataFrame:
    """
    Attach a 'symbol' column if requested, or infer one from the filename.
    """
    if symbol is None:
        symbol = _infer_symbol_from_path(path)

    if symbol is not None:
        df["symbol"] = symbol

    return df


# ---------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------

def load_csv(path: str, symbol: Optional[str] = None) -> Frame:
    """
    Load a CSV with OHLCV data into a Frame.

    Handles:
      - extra header row of symbols (e.g. ",SPY,SPY,SPY,SPY,SPY")
      - adjusted close (if present, uses it as 'close' and stores raw in 'close_raw')
      - datetime parsing and garbage row removal
      - auto symbol detection from filename if symbol is None

    Expected final columns in Frame's pandas view:
      index: datetime
      columns: open, high, low, close, volume, [adj_close?], [close_raw?], [symbol?]
    """

    # Read raw CSV
    df = pd.read_csv(path, header=0)

    # Detect and drop vendor "symbol row":
    #   datetime is empty/NaN and other columns are strings (like 'SPY')
    if "datetime" in df.columns:
        first_dt = df.iloc[0]["datetime"]
        if (pd.isna(first_dt) or str(first_dt).strip() == "") and len(df) > 1:
            df = df.iloc[1:].reset_index(drop=True)

    # Normalize OHLCV + datetime
    df = _normalize_ohlcv_dataframe(df, path=path, prefer_adj_close=True)

    # Attach symbol (explicit or inferred)
    df = _attach_symbol(df, symbol, path)

    # Convert to Frame
    return Frame.from_pandas(df)


# ---------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------

def load_parquet(path: str, symbol: Optional[str] = None) -> Frame:
    """
    Load a Parquet file with OHLCV data into a Frame.

    Handles:
      - case-insensitive column names
      - datetime as index or column
      - adjusted close (same behavior as load_csv)
      - auto symbol detection from filename if symbol is None
    """

    df = pd.read_parquet(path)

    df = _normalize_ohlcv_dataframe(df, path=path, prefer_adj_close=True)
    df = _attach_symbol(df, symbol, path)

    return Frame.from_pandas(df)

def load_directory(
    path: str | Path,
    extensions: Sequence[str] = (".csv", ".parquet"),
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, Frame]:
    """
    Load all OHLCV files in a directory into a dict of {symbol: Frame}.

    - Uses load_csv / load_parquet based on file extension.
    - Auto-detects symbol from filename (e.g. 'SPY_daily.csv' -> 'SPY') unless
      symbol is explicitly passed to the underlying loader.
    - If `symbols` is provided, only load files whose inferred symbol is in that list.

    Example:
        data = load_directory("data")
        # data might contain:
        #   {"SPY": Frame(...), "NDX": Frame(...), "TQQQ": Frame(...), ...}
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{path} is not a directory or does not exist")

    ext_set = {e.lower() for e in extensions}
    symbol_filter = {s.upper() for s in symbols} if symbols is not None else None

    result: Dict[str, Frame] = {}

    for file in sorted(path.iterdir()):
        if not file.is_file():
            continue

        ext = file.suffix.lower()
        if ext not in ext_set:
            continue

        # Infer symbol from filename
        inferred_symbol = _infer_symbol_from_path(str(file))
        if inferred_symbol is None:
            # Skip files we can't infer a symbol from
            continue

        if symbol_filter is not None and inferred_symbol not in symbol_filter:
            continue

        if ext == ".csv":
            frame = load_csv(str(file), symbol=inferred_symbol)
        elif ext == ".parquet":
            frame = load_parquet(str(file), symbol=inferred_symbol)
        else:
            # Shouldn't happen given ext_set, but be safe
            continue

        # Overwrite if duplicate symbol; or you could warn/log instead.
        result[inferred_symbol] = frame

    return result