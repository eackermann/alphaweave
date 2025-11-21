"""Base backtester interface."""

from typing import Protocol, Type
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy
from alphaweave.core.frame import Frame


class BaseBacktester(Protocol):
    """Abstract base class for backtesters."""

    def run(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, Frame],
        capital: float = 100_000.0,
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

