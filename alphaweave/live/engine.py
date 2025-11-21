"""Lightweight live engine wrapper that reuses the vector backtester."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type, Union

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.monitoring.core import Monitor
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy


class LiveEngine:
    """
    Minimal live engine facade.

    For now it reuses the VectorBacktester execution loop but annotates monitor
    metadata with ``mode="live"`` so dashboards can differentiate sources.
    """

    def __init__(
        self,
        *,
        performance_mode: str = "default",
        monitor: Optional[Monitor] = None,
    ):
        self._backtester = VectorBacktester(performance_mode=performance_mode)
        self._monitor = monitor

    def run(
        self,
        strategy_cls: Type[Strategy],
        data: Union[Frame, Dict[str, Frame]],
        capital: float = 100_000.0,
        **kwargs: Any,
    ) -> BacktestResult:
        monitor_override: Optional[Monitor] = kwargs.pop("monitor", None)
        monitor_meta: Optional[Mapping[str, Any]] = kwargs.pop("monitor_meta", None)
        monitor = monitor_override or self._monitor
        live_meta: Dict[str, Any] = {"mode": "live"}
        if monitor_meta:
            live_meta.update(monitor_meta)
        return self._backtester.run(
            strategy_cls,
            data,
            capital=capital,
            monitor=monitor,
            monitor_meta=live_meta,
            **kwargs,
        )


