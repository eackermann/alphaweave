"""Robustness helpers for alphaweave."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from alphaweave.engine.vector import VectorBacktester
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy


@dataclass
class MultiRunResult:
    results: List[BacktestResult]

    @property
    def final_equities(self) -> List[float]:
        return [res.final_equity for res in self.results]

    @property
    def mean_final_equity(self) -> float:
        return float(np.mean(self.final_equities)) if self.results else 0.0

    @property
    def best_result(self) -> Optional[BacktestResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.final_equity)


@dataclass
class SweepEntry:
    params: Dict[str, Any]
    result: BacktestResult


@dataclass
class SweepResult:
    entries: List[SweepEntry]

    def best(self, metric: str = "final_equity", maximize: bool = True) -> Optional[SweepEntry]:
        if not self.entries:
            return None

        def score(entry: SweepEntry) -> float:
            res = entry.result
            value = getattr(res, metric, None)
            if callable(value):
                return float(value())
            if value is None:
                raise AttributeError(f"BacktestResult has no metric '{metric}'")
            return float(value)

        return max(self.entries, key=score) if maximize else min(self.entries, key=score)


@dataclass
class BootstrapResult:
    samples: List[BacktestResult]

    @property
    def final_equities(self) -> List[float]:
        return [res.final_equity for res in self.samples]

    @property
    def mean_final_equity(self) -> float:
        return float(np.mean(self.final_equities)) if self.samples else 0.0


def run_multi_start(
    backtester: VectorBacktester,
    strategy_cls: type[Strategy],
    data,
    runs: int,
    capital: float = 100_000.0,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
) -> MultiRunResult:
    """Run the same strategy multiple times and collect results."""
    results = []
    for _ in range(runs):
        result = backtester.run(
            strategy_cls,
            data=data,
            capital=capital,
            strategy_kwargs=strategy_kwargs,
        )
        results.append(result)
    return MultiRunResult(results=results)


def _run_single_config(args: tuple) -> tuple[Dict[str, Any], BacktestResult]:
    """
    Worker function for parallel parameter sweep.

    Must be top-level for pickling.
    """
    (
        strategy_cls,
        data,
        params,
        capital,
        backtester_kwargs,
    ) = args

    backtester = VectorBacktester(**backtester_kwargs)
    result = backtester.run(
        strategy_cls,
        data=data,
        capital=capital,
        strategy_kwargs=params,
    )
    return params, result


def parameter_sweep(
    backtester: VectorBacktester,
    strategy_cls: type[Strategy],
    data,
    param_grid: Dict[str, Sequence[Any]],
    capital: float = 100_000.0,
    n_jobs: int = 1,
) -> SweepResult:
    """
    Grid search over strategy keyword arguments.

    Args:
        backtester: VectorBacktester instance (used for configuration)
        strategy_cls: Strategy class
        data: Data for backtest
        param_grid: Dictionary mapping parameter names to lists of values
        capital: Starting capital
        n_jobs: Number of parallel jobs (1 = sequential, >1 = parallel)

    Returns:
        SweepResult with all parameter combinations and results
    """
    entries: List[SweepEntry] = []
    keys = list(param_grid.keys())
    grid_values = [list(param_grid[key]) for key in keys]

    # Prepare jobs
    jobs = []
    for combo in product(*grid_values):
        params = dict(zip(keys, combo))
        # Extract backtester configuration
        backtester_kwargs = {
            "performance_mode": getattr(backtester, "performance_mode", "default"),
        }
        jobs.append((strategy_cls, data, params, capital, backtester_kwargs))

    # Execute jobs
    if n_jobs == 1:
        # Sequential execution
        for job in jobs:
            params, result = _run_single_config(job)
            entries.append(SweepEntry(params=params, result=result))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_run_single_config, jobs))
            for params, result in results:
                entries.append(SweepEntry(params=params, result=result))

    return SweepResult(entries=entries)


def bootstrap_equity(
    result: BacktestResult,
    samples: int = 100,
) -> BootstrapResult:
    """Bootstrap the equity curve by resampling returns."""
    equity = result.equity_series
    returns = equity.pct_change().dropna()
    if returns.empty:
        return BootstrapResult(samples=[result])

    start_value = float(equity.iloc[0])
    boot_results: List[BacktestResult] = []

    returns_array = returns.to_numpy()
    for _ in range(samples):
        sampled = np.random.choice(returns_array, size=len(returns_array), replace=True)
        values = [start_value]
        for r in sampled:
            values.append(values[-1] * (1 + r))
        boot_series = pd.Series(values)
        boot_results.append(BacktestResult(boot_series, trades=[]))

    return BootstrapResult(samples=boot_results)
