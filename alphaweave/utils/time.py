"""Time utilities for alphaweave."""

import pandas as pd
from typing import Union


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime column to pandas datetime and set as index if not already.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with datetime index
    """
    df = df.copy()

    # Check if index is already datetime
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Look for datetime column
    datetime_cols = ["datetime", "dt", "timestamp"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            return df

    # If no datetime column found, check if index can be converted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            raise ValueError(
                "DataFrame must have a datetime column or datetime index. "
                f"Columns: {list(df.columns)}, Index type: {type(df.index)}"
            )

    return df

