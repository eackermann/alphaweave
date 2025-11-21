"""Tests for calendar alignment and strategy timestamp helpers."""

from typing import List

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


def make_frame(start: str, periods: int) -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=periods, freq="D"),
            "open": range(periods),
            "high": [v + 1 for v in range(periods)],
            "low": [max(0, v - 1) for v in range(periods)],
            "close": [100 + v for v in range(periods)],
            "volume": [1_000_000] * periods,
        }
    )
    return Frame.from_pandas(df)


class InvestAllStrategy(Strategy):
    def init(self) -> None:
        self.symbols = list(self.data.keys())

    def next(self, index):
        if not self.symbols:
            return
        weight = 1.0 / len(self.symbols)
        for symbol in self.symbols:
            self.order_target_percent(symbol, weight)


class TimestampCaptureStrategy(Strategy):
    timestamps: List[pd.Timestamp] = []

    def init(self) -> None:
        TimestampCaptureStrategy.timestamps = []

    def next(self, index):
        TimestampCaptureStrategy.timestamps.append(self.now())
        self.order_target_percent("ASSET", 1.0)


def test_calendar_alignment_intersection():
    frame_a = make_frame("2020-01-01", 10)  # Jan 1 - Jan 10
    frame_b = make_frame("2020-01-03", 8)   # Jan 3 - Jan 10

    engine = VectorBacktester()
    result = engine.run(
        InvestAllStrategy,
        data={"A": frame_a, "B": frame_b},
        capital=1_000.0,
    )

    assert len(result.equity_series) == 8  # Intersection calendar length


def test_strategy_now_returns_timestamp():
    frame = make_frame("2020-02-01", 5)
    engine = VectorBacktester()
    result = engine.run(
        TimestampCaptureStrategy,
        data={"ASSET": frame},
        capital=1_000.0,
    )

    timestamps = TimestampCaptureStrategy.timestamps
    assert len(timestamps) == len(result.equity_series)
    expected = list(frame.to_pandas().index)
    # now() now returns UTC-aware timestamps
    # Convert expected to UTC-aware for comparison
    expected_utc = [ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC") for ts in expected]
    assert timestamps == expected_utc
