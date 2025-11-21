"""Tests for run_multi_start robustness helper."""

import pandas as pd

from alphaweave.analysis import run_multi_start
from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class AlwaysInvestStrategy(Strategy):
    def init(self) -> None:
        pass

    def next(self, index):
        self.order_target_percent("ASSET", 1.0)


def make_frame() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=20, freq="B"),
            "open": range(20),
            "high": [v + 1 for v in range(20)],
            "low": [max(0, v - 1) for v in range(20)],
            "close": [100 + v for v in range(20)],
            "volume": [1_000_000] * 20,
        }
    )
    return Frame.from_pandas(df)


def test_run_multi_start_collects_results():
    data = {"ASSET": make_frame()}
    engine = VectorBacktester()

    multi_result = run_multi_start(engine, AlwaysInvestStrategy, data=data, runs=3, capital=1_000.0)

    assert len(multi_result.results) == 3
    finals = multi_result.final_equities
    assert len(finals) == 3
    assert multi_result.mean_final_equity == sum(finals) / 3
