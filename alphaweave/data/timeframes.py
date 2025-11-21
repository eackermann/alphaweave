"""Timeframe utilities for alphaweave."""

import pandas as pd

from alphaweave.core.frame import Frame


def resample_frame(frame: Frame, rule: str) -> Frame:
    """Resample a Frame to a new timeframe using OHLCV aggregation."""
    pdf = frame.to_pandas()
    if not isinstance(pdf.index, pd.DatetimeIndex):
        raise ValueError("Frame must have a DatetimeIndex for resampling")

    agg_map = {}
    if "open" in pdf.columns:
        agg_map["open"] = "first"
    if "high" in pdf.columns:
        agg_map["high"] = "max"
    if "low" in pdf.columns:
        agg_map["low"] = "min"
    if "close" in pdf.columns:
        agg_map["close"] = "last"
    if "volume" in pdf.columns:
        agg_map["volume"] = "sum"

    if not agg_map:
        raise ValueError("Frame must contain OHLCV columns for resampling")

    resampled = pdf.resample(rule).agg(agg_map)
    resampled = resampled.dropna(how="all")
    return Frame.from_pandas(resampled)
