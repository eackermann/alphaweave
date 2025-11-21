"""High-level orchestration for live/replay runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Type

from alphaweave.core.frame import Frame
from alphaweave.data.loaders import load_directory
from alphaweave.live.adapters.alpaca import AlpacaAdapter
from alphaweave.live.adapters.base import BrokerAdapter
from alphaweave.live.adapters.binance import BinanceAdapter
from alphaweave.live.adapters.ibkr import IBKRAdapter
from alphaweave.live.adapters.mock import MockBrokerAdapter
from alphaweave.live.adapters.paper import PaperBrokerAdapter
from alphaweave.live.config import LiveConfig
from alphaweave.live.datafeed import DataFeed, ReplayDataFeed
from alphaweave.live.engine import LiveEngine
from alphaweave.live.state import LiveState, save_live_state
from alphaweave.monitoring.core import InMemoryMonitor
from alphaweave.monitoring.dashboard import generate_html_dashboard
from alphaweave.monitoring.run import RunMonitor
from alphaweave.strategy.base import Strategy


BROKER_REGISTRY: Dict[str, Type[BrokerAdapter]] = {
    "mock": MockBrokerAdapter,
    "paper": PaperBrokerAdapter,
    "ibkr": IBKRAdapter,
    "alpaca": AlpacaAdapter,
    "binance": BinanceAdapter,
}


@dataclass
class LiveRunner:
    config: LiveConfig
    broker: BrokerAdapter
    datafeed: DataFeed
    engine: LiveEngine
    monitor: InMemoryMonitor
    strategy_cls: Type[Strategy]
    strategy_params: Mapping[str, Any]

    @classmethod
    def from_config(cls, cfg: LiveConfig) -> "LiveRunner":
        broker_cls = BROKER_REGISTRY.get(cfg.broker.name.lower())
        if broker_cls is None:
            raise ValueError(f"Unknown broker adapter '{cfg.broker.name}'")
        broker = broker_cls.from_config(cfg.broker)

        datafeed = _build_datafeed(cfg.datafeed)
        monitor = InMemoryMonitor()
        engine = LiveEngine(monitor=monitor)
        strategy_cls = _import_strategy(cfg.strategy.class_path)
        strategy_params = cfg.strategy.params or {}
        return cls(
            config=cfg,
            broker=broker,
            datafeed=datafeed,
            engine=engine,
            monitor=monitor,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
        )

    def run(self) -> RunMonitor:
        self.broker.connect()
        self.datafeed.connect()
        try:
            frames = self._materialize_frames()
            capital = float(self.config.datafeed.get("capital", 100_000.0)) if isinstance(self.config.datafeed, Mapping) else 100_000.0
            self._check_risk(capital)
            self.engine.run(
                self.strategy_cls,
                data=frames,
                capital=capital,
                strategy_kwargs=dict(self.strategy_params),
                monitor_meta={"source": "LiveRunner"},
            )
            self._maybe_persist_state()
            self._maybe_write_dashboard()
        finally:
            self.datafeed.disconnect()
            self.broker.disconnect()
        return RunMonitor(self.monitor)

    def _materialize_frames(self) -> Mapping[str, Frame]:
        if isinstance(self.datafeed, ReplayDataFeed):
            return self.datafeed.frames()
        raise NotImplementedError("Only ReplayDataFeed is implemented at the moment")

    def _maybe_persist_state(self) -> None:
        persistence = self.config.persistence
        if not persistence or "state_path" not in persistence:
            return
        account = self.broker.get_account_state()
        state = LiveState(
            timestamp=datetime.now(timezone.utc),
            strategy_state={},  # Placeholder until strategies expose real state
            portfolio_state={
                "cash": account.cash,
                "equity": account.equity,
                "positions": dict(account.positions),
            },
            open_orders=self.broker.get_open_orders(),
        )
        save_live_state(persistence["state_path"], state)

    def _maybe_write_dashboard(self) -> None:
        monitor_cfg = self.config.monitor or {}
        html_path = monitor_cfg.get("dashboard_html")
        if not html_path:
            return
        title = monitor_cfg.get("dashboard_title", "Live Run Dashboard")
        run = RunMonitor(self.monitor)
        html = generate_html_dashboard(run, title=title)
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        Path(html_path).write_text(html, encoding="utf-8")

    def _check_risk(self, capital: float) -> None:
        risk_cfg = self.config.risk or {}
        max_capital = risk_cfg.get("max_capital")
        if max_capital is not None and capital > float(max_capital):
            raise ValueError(f"Proposed capital {capital} exceeds max_capital {max_capital}")


def _build_datafeed(cfg: Mapping[str, Any]) -> DataFeed:
    feed_type = cfg.get("type", "replay").lower()
    if feed_type == "replay":
        if "path" in cfg:
            frames = load_directory(cfg["path"], symbols=cfg.get("symbols"))
        elif "frames" in cfg:
            frames = cfg["frames"]
        else:
            raise ValueError("Replay datafeed requires 'path' or 'frames'")
        return ReplayDataFeed(frames)
    raise NotImplementedError(f"Datafeed type '{feed_type}' not implemented")


def _import_strategy(path: str) -> Type[Strategy]:
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid strategy class path '{path}'")
    module = import_module(module_name)
    strategy_cls = getattr(module, class_name)
    if not issubclass(strategy_cls, Strategy):
        raise TypeError(f"{path} is not a Strategy subclass")
    return strategy_cls


