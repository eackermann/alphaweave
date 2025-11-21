"""Tests for parameter_sweep helper."""

import pandas as pd

from alphaweave.analysis import parameter_sweep
from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class TargetStrategy(Strategy):
    def __init__(self, data, target: float = 1.0):
        super().__init__(data)
        self.target = target

    def init(self) -> None:
        pass

    def next(self, index):
        self.order_target_percent("ASSET", self.target)


def make_frame() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=30, freq="B"),
            "open": range(30),
            "high": [v + 1 for v in range(30)],
            "low": [max(0, v - 1) for v in range(30)],
            "close": [100 + v for v in range(30)],
            "volume": [500_000] * 30,
        }
    )
    return Frame.from_pandas(df)


def test_parameter_sweep_identifies_best_params():
    data = {"ASSET": make_frame()}
    engine = VectorBacktester()

    sweep = parameter_sweep(
        engine,
        TargetStrategy,
        data=data,
        param_grid={"target": [0.0, 1.0]},
        capital=1_000.0,
    )

    assert len(sweep.entries) == 2
    best_entry = sweep.best(metric="final_equity")
    assert best_entry is not None
    assert best_entry.params["target"] == 1.0
