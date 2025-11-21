"""Tests for timeframe resampling and multi-timeframe helpers."""

from typing import List

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.data.timeframes import resample_frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


def make_daily_frame(start: str, periods: int) -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=periods, freq="D"),
            "open": [10 + i for i in range(periods)],
            "high": [11 + i for i in range(periods)],
            "low": [9 + i for i in range(periods)],
            "close": [10 + (i % 5) + i for i in range(periods)],
            "volume": [1_000 + i for i in range(periods)],
        }
    )
    return Frame.from_pandas(df)


def test_resample_frame_weekly_ohlcv():
    frame = make_daily_frame("2020-01-01", 7)
    weekly = resample_frame(frame, "W")
    pdf = weekly.to_pandas()
    original = frame.to_pandas()
    first_label = pdf.index[0]
    subset = original.loc[:first_label]
    first_row = pdf.iloc[0]

    assert first_row["open"] == subset["open"].iloc[0]
    assert first_row["high"] == subset["high"].max()
    assert first_row["low"] == subset["low"].min()
    assert first_row["close"] == subset["close"].iloc[-1]
    assert first_row["volume"] == subset["volume"].sum()


class MultiTimeframeStrategy(Strategy):
    pairs: List[tuple] = []

    def init(self) -> None:
        MultiTimeframeStrategy.pairs = []
        self.register_timeframe("1W", "W")

    def next(self, index):
        daily = self.close("X")
        try:
            weekly = self.close("X", timeframe="1W")
        except ValueError:
            weekly = None
        MultiTimeframeStrategy.pairs.append((daily, weekly))
        self.order_target_percent("X", 1.0)


def test_multi_timeframe_close_behavior():
    frame = make_daily_frame("2020-01-01", 20)
    data = {"X": frame}
    engine = VectorBacktester()
    result = engine.run(MultiTimeframeStrategy, data=data, capital=1_000.0)

    pairs = MultiTimeframeStrategy.pairs
    assert len(pairs) == len(result.equity_series)

    daily_index = frame.to_pandas().index
    weekly_series = resample_frame(frame, "W").to_pandas()["close"]

    expected_weekly = []
    for ts in daily_index:
        # Ensure timezone compatibility
        if ts.tz is not None and weekly_series.index.tz is None:
            ts = ts.tz_localize(None)
        elif ts.tz is None and weekly_series.index.tz is not None:
            ts = ts.tz_localize("UTC")
        eligible = weekly_series.loc[:ts]
        if eligible.empty:
            expected_weekly.append(None)
        else:
            expected_weekly.append(float(eligible.iloc[-1]))

    weekly_values = [w for _, w in pairs]
    assert weekly_values == expected_weekly
