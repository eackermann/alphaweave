"""Tests for live stack scaffolding."""

from pathlib import Path

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.core.types import Order, OrderType
from alphaweave.live.adapters.mock import MockBrokerAdapter
from alphaweave.live.config import LiveConfig, StrategyConfig, parse_live_config
from alphaweave.live.runner import LiveRunner
from alphaweave.strategy.base import Strategy


class LiveTestStrategy(Strategy):
    def init(self):
        self.symbols = list(self.data.keys())

    def next(self, i):
        for symbol in self.symbols:
            self.order_target_percent(symbol, 0.5)


def _make_frame() -> Frame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    return Frame.from_pandas(df)


def test_mock_broker_submit_order_updates_portfolio():
    broker = MockBrokerAdapter(initial_cash=10_000.0, default_price=100.0)
    broker.connect()
    broker.update_prices({"TEST": 100.0})
    order = Order(symbol="TEST", size=10.0, order_type=OrderType.MARKET)
    broker.submit_order(order)
    account = broker.get_account_state()
    assert account.positions["TEST"] == 10.0
    assert broker.poll_fills(), "fills should accumulate until polled"
    broker.disconnect()


def test_live_runner_replay(tmp_path: Path):
    frame = _make_frame()
    cfg = parse_live_config(
        {
            "broker": {"name": "mock"},
            "strategy": {"class": "tests.test_live_stack.LiveTestStrategy"},
            "datafeed": {"type": "replay", "frames": {"TEST": frame}, "capital": 50_000.0},
            "monitor": {"dashboard_html": tmp_path / "dash.html"},
            "persistence": {"state_path": tmp_path / "state.pkl"},
        }
    )
    runner = LiveRunner.from_config(cfg)
    run = runner.run()
    assert not run.bars.empty
    assert (tmp_path / "dash.html").exists()
    assert (tmp_path / "state.pkl").exists()


