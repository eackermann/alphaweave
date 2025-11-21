"""Helpers for persisting live trading state."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class LiveState:
    timestamp: datetime
    strategy_state: Mapping[str, object]
    portfolio_state: Mapping[str, object]
    open_orders: Sequence[Mapping[str, object]]


def save_live_state(path: str | Path, state: LiveState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(state, handle)


def load_live_state(path: str | Path) -> LiveState:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


