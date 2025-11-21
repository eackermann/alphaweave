"""Tests for walk-forward optimization."""

from typing import Iterable

import numpy as np
import pandas as pd

from alphaweave.analysis import walk_forward_optimize
from alphaweave.core.frame import Frame
from alphaweave.strategy.base import Strategy


class LookbackStrategy(Strategy):
    def __init__(self, data, lookback: int = 2):
        super().__init__(data)
        self.lookback = lookback

    def init(self) -> None:
        self.prices = self.series("ASSET", field="close")

    def next(self, index):
        if index < self.lookback:
            return
        if self.prices.iloc[index] > self.prices.iloc[index - self.lookback]:
            self.order_target_percent("ASSET", 1.0)
        else:
            self.order_target_percent("ASSET", 0.0)


class ExposureStrategy(Strategy):
    def __init__(self, data, exposure: float = 1.0):
        super().__init__(data)
        self.exposure = exposure

    def init(self) -> None:
        pass

    def next(self, index):
        self.order_target_percent("ASSET", self.exposure)


def make_frame(prices: Iterable[float]) -> Frame:
    prices = list(prices)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=len(prices), freq="B"),
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        }
    )
    return Frame.from_pandas(df)


def test_walkforward_basic_outputs():
    prices = [100 + i + (i % 5 - 2) for i in range(60)]
    data = {"ASSET": make_frame(prices)}

    wf = walk_forward_optimize(
        LookbackStrategy,
        data=data,
        capital=1_000.0,
        train_window=10,
        test_window=5,
        param_grid={"lookback": [2, 4]},
    )

    assert wf.windows
    metrics = wf.metrics_frame()
    assert len(metrics) == len(wf.windows)
    expected_cols = {
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "metric_value",
        "test_final_equity",
        "test_total_return",
        "test_max_drawdown",
        "test_sharpe",
        "lookback",
    }
    assert expected_cols.issubset(metrics.columns)

    total_test_len = sum(len(window.test_result.equity_series) for window in wf.windows)
    assert len(wf.oos_equity_series()) == total_test_len


def test_metric_string_vs_callable():
    prices = [100 + (i % 7) + i * 0.5 for i in range(40)]
    data = {"ASSET": make_frame(prices)}

    wf_attr = walk_forward_optimize(
        LookbackStrategy,
        data=data,
        capital=500.0,
        train_window=8,
        test_window=4,
        param_grid={"lookback": [2, 3]},
        metric="total_return",
    )

    wf_callable = walk_forward_optimize(
        LookbackStrategy,
        data=data,
        capital=500.0,
        train_window=8,
        test_window=4,
        param_grid={"lookback": [2, 3]},
        metric=lambda r: r.total_return,
    )

    np.testing.assert_allclose(
        wf_attr.metrics_frame()["metric_value"],
        wf_callable.metrics_frame()["metric_value"],
    )


def test_max_drawdown_metric_prefers_lower_drawdown():
    prices = [100, 110, 90, 95, 92, 91, 93, 94, 90, 92, 91, 93]
    data = {"ASSET": make_frame(prices)}

    wf = walk_forward_optimize(
        ExposureStrategy,
        data=data,
        capital=1_000.0,
        train_window=4,
        test_window=3,
        param_grid={"exposure": [0.0, 1.0]},
        metric="max_drawdown",
    )

    metrics = wf.metrics_frame()
    assert metrics["exposure"].isin([0.0]).all()
