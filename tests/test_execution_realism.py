"""Tests for execution realism features."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.risk import RiskLimits
from alphaweave.engine.vector import VectorBacktester
from alphaweave.execution.volume import VolumeLimitModel
from alphaweave.strategy.base import Strategy


def make_frame(prices, volumes=None):
    prices = list(prices)
    if volumes is None:
        volumes = [1_000] * len(prices)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=len(prices), freq="D"),
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )
    return Frame.from_pandas(df)


class MarketOrderStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)
        self.fired = False

    def next(self, index):
        if not self.fired:
            self.order_market("X", 10_000)
            self.fired = True


class TargetStrategy(Strategy):
    def next(self, index):
        self.order_target_percent("A", 1.0)
        self.order_target_percent("B", 1.0)


class ConditionalOrdersStrategy(Strategy):
    def next(self, index):
        self.order_limit("X", 1.0, limit_price=95.0)
        self.order_limit("X", 1.0, limit_price=85.0)
        self.order_stop("X", 1.0, stop_price=108.0)
        self.order_stop("X", -1.0, stop_price=92.0)


def test_volume_limit_clamps_market_orders():
    frame = make_frame([100, 101, 102], volumes=[1_000, 1_000, 1_000])
    engine = VectorBacktester()
    volume_model = VolumeLimitModel(max_pct_volume=0.1)
    result = engine.run(
        MarketOrderStrategy,
        data={"X": frame},
        capital=10_000.0,
        volume_limit=volume_model,
    )
    assert result.trades
    fill = result.trades[0]
    assert abs(fill.size - 100.0) < 1e-9


def test_risk_limits_scale_target_weights():
    frame = make_frame([100, 100, 100])
    engine = VectorBacktester()
    limits = RiskLimits(max_symbol_weight=0.75, max_gross_leverage=1.0)
    result = engine.run(
        TargetStrategy,
        data={"A": frame, "B": frame},
        capital=1_000.0,
        risk_limits=limits,
    )
    sizes = sorted(fill.size for fill in result.trades)
    assert len(sizes) == 2
    assert all(abs(size - 5.0) < 1e-6 for size in sizes)


def test_limit_and_stop_order_fills():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=1, freq="D"),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1_000.0],
        }
    )
    frame = Frame.from_pandas(df)
    engine = VectorBacktester()
    result = engine.run(
        ConditionalOrdersStrategy,
        data={"X": frame},
        capital=10_000.0,
    )
    assert len(result.trades) == 3
    prices = sorted(fill.price for fill in result.trades)
    assert prices == [92.0, 95.0, 108.0]
