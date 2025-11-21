"""Tests for monitoring infrastructure."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.monitoring.core import BarSnapshot, InMemoryMonitor, TradeRecord
from alphaweave.monitoring.run import RunMonitor
from alphaweave.strategy.base import Strategy


class MonitorTestStrategy(Strategy):
    def init(self):
        pass

    def next(self, i):
        self.order_target_percent("TEST", 0.5)
        self.log_metric("bar_index", float(i))


def _make_frame(periods: int = 5) -> Frame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + i for i in range(periods)],
            "high": [101.0 + i for i in range(periods)],
            "low": [99.0 + i for i in range(periods)],
            "close": [100.0 + i for i in range(periods)],
            "volume": [1000] * periods,
        }
    )
    return Frame.from_pandas(df)


def test_inmemory_monitor_records_data():
    monitor = InMemoryMonitor()
    now = datetime.now(UTC)
    monitor.on_run_start(meta={"mode": "test"})
    monitor.on_bar(
        BarSnapshot(
            timestamp=now,
            prices={"A": 100.0},
            equity=1000.0,
            cash=500.0,
            positions_value=500.0,
            positions={"A": 5.0},
            leverage=1.0,
            pnl=10.0,
            drawdown=0.0,
        )
    )
    monitor.on_trade(
        TradeRecord(
            timestamp=now,
            symbol="A",
            side="buy",
            price=100.0,
            size=5.0,
            fees=0.5,
        )
    )
    monitor.on_metric("alpha", 1.23, now)
    monitor.on_run_end()

    bars_df = monitor.bars_df()
    trades_df = monitor.trades_df()
    metrics_df = monitor.metrics_df()

    assert len(bars_df) == 1
    assert bars_df.iloc[0]["equity"] == 1000.0
    assert len(trades_df) == 1
    assert trades_df.iloc[0]["symbol"] == "A"
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["name"] == "alpha"


def test_run_monitor_derivatives():
    monitor = InMemoryMonitor()
    t0 = datetime.now(UTC)
    monitor.on_run_start()
    monitor.on_bar(
        BarSnapshot(
            timestamp=t0,
            prices={"A": 100.0},
            equity=1000.0,
            cash=600.0,
            positions_value=400.0,
            positions={"A": 4.0},
        )
    )
    monitor.on_bar(
        BarSnapshot(
            timestamp=t0 + timedelta(minutes=1),
            prices={"A": 102.0},
            equity=1010.0,
            cash=500.0,
            positions_value=510.0,
            positions={"A": 5.0},
        )
    )
    monitor.on_trade(
        TradeRecord(
            timestamp=t0,
            symbol="A",
            side="buy",
            price=100.0,
            size=4.0,
            fees=0.1,
            slippage=0.0,
        )
    )
    monitor.on_trade(
        TradeRecord(
            timestamp=t0 + timedelta(minutes=1),
            symbol="A",
            side="buy",
            price=102.0,
            size=1.0,
            fees=0.05,
            slippage=0.0,
        )
    )
    monitor.on_run_end()

    run = RunMonitor(monitor)
    exposure = run.exposure_over_time()
    turnover = run.turnover_over_time()
    cost = run.cost_over_time()

    assert not exposure.empty
    assert turnover.iloc[-1] >= 0.0
    assert cost.iloc[-1] > 0.0


def test_vector_backtester_emits_monitor_events():
    frame = _make_frame()
    monitor = InMemoryMonitor()
    backtester = VectorBacktester()

    backtester.run(
        MonitorTestStrategy,
        data={"TEST": frame},
        capital=10000.0,
        monitor=monitor,
    )

    assert len(monitor.bars) == len(frame.to_pandas())
    assert not monitor.trades_df().empty
    metrics = monitor.metrics_df()
    assert "bar_index" in metrics["name"].unique()
    assert monitor.meta.get("mode") == "backtest"


