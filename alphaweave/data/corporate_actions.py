"""Corporate actions (splits and dividends) support for alphaweave."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from alphaweave.core.frame import Frame


@dataclass
class SplitAction:
    """Represents a stock split action."""

    symbol: str
    date: datetime
    ratio: float  # e.g., 2.0 for a 2-for-1 split

    def __repr__(self) -> str:
        return f"SplitAction(symbol={self.symbol}, date={self.date}, ratio={self.ratio})"


@dataclass
class DividendAction:
    """Represents a cash dividend action."""

    symbol: str
    date: datetime
    amount: float  # dividend per share

    def __repr__(self) -> str:
        return f"DividendAction(symbol={self.symbol}, date={self.date}, amount={self.amount})"


class CorporateActionsStore:
    """Stores and provides access to corporate actions by symbol and date."""

    def __init__(self):
        # Map symbol -> list of SplitAction, sorted by date
        self._splits: Dict[str, List[SplitAction]] = {}
        # Map symbol -> list of DividendAction, sorted by date
        self._dividends: Dict[str, List[DividendAction]] = {}

    def add_split(self, split: SplitAction) -> None:
        """Add a split action."""
        if split.symbol not in self._splits:
            self._splits[split.symbol] = []
        self._splits[split.symbol].append(split)
        # Keep sorted by date
        self._splits[split.symbol].sort(key=lambda x: x.date)

    def add_dividend(self, dividend: DividendAction) -> None:
        """Add a dividend action."""
        if dividend.symbol not in self._dividends:
            self._dividends[dividend.symbol] = []
        self._dividends[dividend.symbol].append(dividend)
        # Keep sorted by date
        self._dividends[dividend.symbol].sort(key=lambda x: x.date)

    def get_splits_on_date(self, symbol: str, date: datetime) -> List[SplitAction]:
        """Get all splits for a symbol on a specific date."""
        splits = self._splits.get(symbol, [])
        return [s for s in splits if s.date.date() == date.date()]

    def get_dividends_on_date(self, symbol: str, date: datetime) -> List[DividendAction]:
        """Get all dividends for a symbol on a specific date."""
        dividends = self._dividends.get(symbol, [])
        return [d for d in dividends if d.date.date() == date.date()]

    def has_actions_for_symbol(self, symbol: str) -> bool:
        """Check if there are any corporate actions for a symbol."""
        return symbol in self._splits or symbol in self._dividends


def load_splits_csv(path: str) -> List[SplitAction]:
    """
    Load split actions from a CSV file.

    Expected CSV format:
        symbol,date,ratio
        AAPL,2020-08-31,4.0
        MSFT,2003-02-18,2.0

    Where:
        - symbol: stock symbol
        - date: split date (YYYY-MM-DD format)
        - ratio: split ratio (e.g., 2.0 for 2-for-1, 4.0 for 4-for-1)
    """
    df = pd.read_csv(path)
    
    # Normalize column names (case-insensitive)
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("symbol", "sym", "ticker"):
            rename_map[col] = "symbol"
        elif lc in ("date", "datetime", "timestamp"):
            rename_map[col] = "date"
        elif lc in ("ratio", "split_ratio", "split"):
            rename_map[col] = "ratio"
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Validate required columns
    required = ["symbol", "date", "ratio"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    # Parse ratio
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df.dropna(subset=["ratio"])
    
    # Validate ratio > 0
    if (df["ratio"] <= 0).any():
        raise ValueError("Split ratio must be positive")
    
    splits = []
    for _, row in df.iterrows():
        split = SplitAction(
            symbol=str(row["symbol"]).upper(),
            date=row["date"].to_pydatetime(),
            ratio=float(row["ratio"]),
        )
        splits.append(split)
    
    return splits


def load_dividends_csv(path: str) -> List[DividendAction]:
    """
    Load dividend actions from a CSV file.

    Expected CSV format:
        symbol,date,amount
        AAPL,2023-11-16,0.24
        MSFT,2023-11-15,0.75

    Where:
        - symbol: stock symbol
        - date: dividend payment date (YYYY-MM-DD format)
        - amount: dividend per share
    """
    df = pd.read_csv(path)
    
    # Normalize column names (case-insensitive)
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("symbol", "sym", "ticker"):
            rename_map[col] = "symbol"
        elif lc in ("date", "datetime", "timestamp"):
            rename_map[col] = "date"
        elif lc in ("amount", "dividend", "div", "dividend_per_share"):
            rename_map[col] = "amount"
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Validate required columns
    required = ["symbol", "date", "amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    # Parse amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    
    dividends = []
    for _, row in df.iterrows():
        dividend = DividendAction(
            symbol=str(row["symbol"]).upper(),
            date=row["date"].to_pydatetime(),
            amount=float(row["amount"]),
        )
        dividends.append(dividend)
    
    return dividends


def build_corporate_actions_store(
    splits: Optional[List[SplitAction]] = None,
    dividends: Optional[List[DividendAction]] = None,
) -> CorporateActionsStore:
    """
    Build a CorporateActionsStore from lists of splits and dividends.

    Args:
        splits: Optional list of SplitAction objects
        dividends: Optional list of DividendAction objects

    Returns:
        CorporateActionsStore containing all provided actions
    """
    store = CorporateActionsStore()
    
    if splits:
        for split in splits:
            store.add_split(split)
    
    if dividends:
        for dividend in dividends:
            store.add_dividend(dividend)
    
    return store


def detect_split_adjustment_for_symbol(
    frame: Frame,
    splits: List[SplitAction],
    price_column: str = "close",
    lookback_days: int = 5,
    tolerance: float = 0.25,
) -> Tuple[Optional[bool], pd.DataFrame]:
    """
    Heuristically determine whether a price series in `frame` is already
    adjusted for the given `splits`.

    Returns (is_adjusted, diagnostics_df):

      - is_adjusted:
          True  => prices look already split-adjusted (no discontinuity)
          False => prices look raw (visible ~ratio jump at split)
          None  => insufficient data / inconclusive

      - diagnostics_df:
          DataFrame with one row per split, containing:
              'date'         : split date
              'ratio'        : declared split ratio from SplitAction
              'p_before'     : price before split (NaN if unavailable)
              'p_after'      : price after split (NaN if unavailable)
              'ratio_est'    : p_before / p_after
              'delta_to_1'   : |ratio_est - 1|
              'delta_to_ratio': |ratio_est - ratio|
    """
    pdf = frame.to_pandas().copy()
    if not isinstance(pdf.index, pd.DatetimeIndex):
        raise ValueError("Frame index must be a DatetimeIndex for split detection.")

    # 🔧 NEW: ensure index is tz-naive
    if pdf.index.tz is not None:
        pdf.index = pdf.index.tz_convert(None)

    if price_column not in pdf.columns:
        raise ValueError(f"Column '{price_column}' not found in frame.")

    pdf = pdf.sort_index()
    records = []

    for s in splits:
        # 🔧 NEW: normalize split_date to tz-naive midnight
        split_date = pd.Timestamp(s.date)
        if split_date.tzinfo is not None:
            split_date = split_date.tz_convert(None)
        split_date = split_date.normalize()

        # previous bar within lookback window
        before_mask = (pdf.index < split_date) & (
            pdf.index >= split_date - pd.Timedelta(days=lookback_days)
        )
        after_mask = (pdf.index >= split_date) & (
            pdf.index <= split_date + pd.Timedelta(days=lookback_days)
        )

        if not before_mask.any() or not after_mask.any():
            records.append({
                "date": split_date,
                "ratio": s.ratio,
                "p_before": float("nan"),
                "p_after": float("nan"),
                "ratio_est": float("nan"),
                "delta_to_1": float("nan"),
                "delta_to_ratio": float("nan"),
            })
            continue

        p_before = pdf.loc[before_mask, price_column].iloc[-1]
        p_after = pdf.loc[after_mask, price_column].iloc[0]

        if p_after == 0 or pd.isna(p_before) or pd.isna(p_after):
            ratio_est = float("nan")
        else:
            ratio_est = float(p_before) / float(p_after)

        delta_to_1 = abs(ratio_est - 1.0) if pd.notna(ratio_est) else float("nan")
        delta_to_ratio = abs(ratio_est - float(s.ratio)) if pd.notna(ratio_est) else float("nan")

        records.append({
            "date": split_date,
            "ratio": s.ratio,
            "p_before": p_before,
            "p_after": p_after,
            "ratio_est": ratio_est,
            "delta_to_1": delta_to_1,
            "delta_to_ratio": delta_to_ratio,
        })

    diag = pd.DataFrame.from_records(records)

    valid = diag.dropna(subset=["ratio_est"])
    if valid.empty:
        return None, diag

    adjusted_like = (valid["delta_to_1"] < tolerance).sum()
    raw_like = (valid["delta_to_ratio"] < tolerance).sum()

    if adjusted_like == 0 and raw_like == 0:
        return None, diag

    if adjusted_like > raw_like:
        return True, diag
    elif raw_like > adjusted_like:
        return False, diag
    else:
        return None, diag



def auto_filter_splits_for_data(
    data: Dict[str, Frame],
    splits_by_symbol: Dict[str, List[SplitAction]],
    price_column: str = "close",
    lookback_days: int = 5,
    tolerance: float = 0.25,
) -> Dict[str, List[SplitAction]]:
    """
    For each symbol in `splits_by_symbol`, detect whether the corresponding
    `data[symbol]` series is already split-adjusted. If so, drop its splits.

    Returns a new dict: symbol -> filtered list of SplitAction.
    """
    filtered: Dict[str, List[SplitAction]] = {}
    for sym, splits in splits_by_symbol.items():
        frame = data.get(sym)
        if frame is None or not splits:
            filtered[sym] = splits
            continue

        is_adjusted, _diag = detect_split_adjustment_for_symbol(
            frame,
            splits,
            price_column=price_column,
            lookback_days=lookback_days,
            tolerance=tolerance,
        )

        if is_adjusted is True:
            # Already adjusted → do not apply splits again
            filtered[sym] = []
        else:
            filtered[sym] = splits

    return filtered


def build_corporate_actions_store_auto(
    data: Dict[str, Frame],
    splits: List[SplitAction],
    dividends: List[DividendAction],
    price_column: str = "close",
    lookback_days: int = 5,
    tolerance: float = 0.25,
) -> CorporateActionsStore:
    """
    Build a CorporateActionsStore but automatically drop splits for symbols
    whose price series appears already split-adjusted.

    Dividends are left unchanged.

    Args:
        data: Dictionary mapping symbol names to Frame objects
        splits: List of all SplitAction objects
        dividends: List of all DividendAction objects
        price_column: Column name to use for price analysis (default: "close")
        lookback_days: Number of days to look back/forward around split date
        tolerance: Tolerance for ratio comparison (default: 0.25)

    Returns:
        CorporateActionsStore with filtered splits
    """
    # group splits by symbol
    split_map: Dict[str, List[SplitAction]] = {}
    for s in splits:
        split_map.setdefault(s.symbol, []).append(s)

    # filter splits based on data
    filtered_splits_map = auto_filter_splits_for_data(
        data=data,
        splits_by_symbol=split_map,
        price_column=price_column,
        lookback_days=lookback_days,
        tolerance=tolerance,
    )

    # flatten back into list
    filtered_splits: List[SplitAction] = []
    for sym, lst in filtered_splits_map.items():
        filtered_splits.extend(lst)

    return build_corporate_actions_store(
        splits=filtered_splits,
        dividends=dividends,
    )

