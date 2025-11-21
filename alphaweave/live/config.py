"""Live trading configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from alphaweave.live.adapters.base import BrokerConfig

try:  # optional dependency
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore


@dataclass
class StrategyConfig:
    class_path: str
    params: Mapping[str, Any] | None = None


@dataclass
class LiveConfig:
    broker: BrokerConfig
    strategy: StrategyConfig
    datafeed: Mapping[str, Any]
    risk: Mapping[str, Any] | None = None
    monitor: Mapping[str, Any] | None = None
    persistence: Mapping[str, Any] | None = None


def load_live_config(path: str | Path) -> LiveConfig:
    """Load config from YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError("Config file must contain a mapping at the root")

    return parse_live_config(data)


def parse_live_config(raw: Mapping[str, Any]) -> LiveConfig:
    """Parse LiveConfig from a Python mapping (useful for tests)."""
    broker = raw.get("broker")
    strategy = raw.get("strategy")
    datafeed = raw.get("datafeed")
    if broker is None or strategy is None or datafeed is None:
        raise ValueError("Config must include broker/strategy/datafeed sections")

    broker_cfg = BrokerConfig(
        name=str(broker["name"]),
        account_id=broker.get("account_id"),
        extra=broker.get("extra"),
    )
    strategy_cfg = StrategyConfig(
        class_path=str(strategy["class"]),
        params=strategy.get("params"),
    )
    return LiveConfig(
        broker=broker_cfg,
        strategy=strategy_cfg,
        datafeed=datafeed,
        risk=raw.get("risk"),
        monitor=raw.get("monitor"),
        persistence=raw.get("persistence"),
    )


