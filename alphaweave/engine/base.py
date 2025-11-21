"""Base backtester interface."""

from typing import Protocol, Type, Optional

from alphaweave.core.frame import Frame
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy

try:  # Optional import to avoid circular dependency during typing
    from alphaweave.monitoring.core import Monitor
except ImportError:  # pragma: no cover - monitoring optional at import time
    Monitor = object  # type: ignore


class BaseBacktester(Protocol):
    """Abstract base class for backtesters."""

    def run(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, Frame],
        capital: float = 100_000.0,
        monitor: Optional["Monitor"] = None,
    ) -> BacktestResult:
        """
        Run the backtest and return BacktestResult.

        Args:
            strategy_cls: Strategy class to instantiate
            data: Dictionary mapping symbol names to Frame objects
            capital: Starting capital

        Returns:
            BacktestResult with equity series and trades
        """
        raise NotImplementedError

