"""Tests for backtester execution with fees and slippage."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.execution.fees import PerShareFees
from alphaweave.execution.slippage import FixedBpsSlippage
from alphaweave.strategy.base import Strategy


class AlwaysLong(Strategy):
    """Strategy that stays fully invested."""

    def init(self) -> None:
        pass

    def next(self, index):
        self.order_target_percent("ASSET", 1.0)


def make_data() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=50, freq="B"),
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100 + i for i in range(50)],
            "volume": [1_000_000] * 50,
        }
    )
    return Frame.from_pandas(df)


def test_backtester_with_and_without_costs():
    frame = make_data()
    data = {"ASSET": frame}
    engine = VectorBacktester()

    result_no_costs = engine.run(AlwaysLong, data=data, capital=10_000.0)

    result_with_costs = engine.run(
        AlwaysLong,
        data=data,
        capital=10_000.0,
        fees=PerShareFees(0.01),
        slippage=FixedBpsSlippage(25),
    )

    assert result_with_costs.final_equity < result_no_costs.final_equity
