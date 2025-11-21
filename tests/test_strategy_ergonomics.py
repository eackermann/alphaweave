"""Tests for Strategy ergonomics helpers."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.indicators.sma import SMA
from alphaweave.strategy.base import Strategy


class ErgonomicStrategy(Strategy):
    """Strategy that exercises helper methods."""

    last_instance: "ErgonomicStrategy" | None = None

    def __init__(self, data):
        super().__init__(data)
        self.close_values: list[float] = []
        self.sma_values: list[float] = []
        self.series_snapshot = None
        self._series_captured = False
        ErgonomicStrategy.last_instance = self

    def init(self) -> None:  # pragma: no cover - nothing to init
        pass

    def next(self, index):
        close_value = self.close("ASSET")
        self.close_values.append(close_value)

        if not self._series_captured:
            self.series_snapshot = self.series("ASSET", field="close")
            self._series_captured = True

        self.sma_values.append(self.sma("ASSET", period=3))


def make_frame() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100 + i for i in range(10)],
            "volume": [1_000_000] * 10,
        }
    )
    return Frame.from_pandas(df)


def test_strategy_helpers_agree_with_indicators():
    frame = make_frame()
    data = {"ASSET": frame}
    engine = VectorBacktester()

    result = engine.run(ErgonomicStrategy, data=data, capital=1_000.0)
    assert result is not None

    strategy = ErgonomicStrategy.last_instance
    assert strategy is not None

    close_series = frame.to_pandas()["close"]

    # close() should match the raw close price for each bar
    assert strategy.close_values == close_series.tolist()

    # series() should match the pandas Series directly
    assert strategy.series_snapshot.equals(close_series)

    # SMA helper should match direct indicator values
    expected_sma = SMA(close_series, period=3)
    expected_values = [float(expected_sma[i]) for i in range(len(close_series))]
    assert len(strategy.sma_values) == len(expected_values)
    for got, expected in zip(strategy.sma_values, expected_values):
        assert abs(got - expected) < 1e-10
