"""Walk-forward optimization helpers for alphaweave."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy


@dataclass
class WalkForwardWindowResult:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: Dict[str, Any]
    metric_value: float
    train_result: BacktestResult
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    windows: List[WalkForwardWindowResult]

    def metrics_frame(self) -> pd.DataFrame:
        if not self.windows:
            return pd.DataFrame(
                columns=[
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                    "metric_value",
                    "test_final_equity",
                    "test_total_return",
                    "test_max_drawdown",
                    "test_sharpe",
                ]
            )

        param_keys = sorted({key for w in self.windows for key in w.best_params.keys()})
        rows: List[Dict[str, Any]] = []
        for window in self.windows:
            row = {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "metric_value": window.metric_value,
                "test_final_equity": window.test_result.final_equity,
                "test_total_return": window.test_result.total_return,
                "test_max_drawdown": window.test_result.max_drawdown,
                "test_sharpe": window.test_result.sharpe(),
            }
            for key in param_keys:
                row[key] = window.best_params.get(key)
            rows.append(row)
        return pd.DataFrame(rows)

    def oos_equity_series(self) -> pd.Series:
        if not self.windows:
            return pd.Series(dtype=float)
        segments = [w.test_result.equity_series.reset_index(drop=True) for w in self.windows]
        return pd.concat(segments, ignore_index=True)


def walk_forward_optimize(
    strategy_cls: Type[Strategy],
    data: Union[Frame, Dict[str, Frame]],
    capital: float,
    train_window: int,
    test_window: int,
    param_grid: Dict[str, Sequence[Any]],
    metric: Union[str, Callable[[BacktestResult], float]] = "sharpe",
    step: Optional[int] = None,
    strategy_base_kwargs: Optional[Dict[str, Any]] = None,
) -> WalkForwardResult:
    """Run a walk-forward optimization over the provided data."""

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")

    if step is None:
        step = test_window
    if step <= 0:
        raise ValueError("step must be positive")

    if isinstance(data, Frame):
        data_dict: Dict[str, Frame] = {"_default": data}
    elif isinstance(data, dict):
        if not data:
            raise ValueError("data must contain at least one symbol")
        data_dict = data
    else:
        raise TypeError("data must be a Frame or dict[str, Frame]")

    data_frames = {symbol: frame.to_pandas() for symbol, frame in data_dict.items()}
    bar_index = next(iter(data_frames.values())).index
    n_bars = len(bar_index)

    grid_keys = list(param_grid.keys())
    grid_values = [list(param_grid[key]) for key in grid_keys]
    if not grid_keys:
        grid_values = [[]]

    base_kwargs = strategy_base_kwargs.copy() if strategy_base_kwargs else {}
    backtester = VectorBacktester()

    windows: List[WalkForwardWindowResult] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_bars:
            break

        train_data = _slice_data(data_frames, train_start, train_end)
        test_data = _slice_data(data_frames, test_start, test_end)
        if train_data is None or test_data is None:
            break

        best_params = None
        best_metric_value = None
        best_score = -np.inf
        best_train_result = None

        combos = product(*grid_values) if grid_keys else [()]
        for combo in combos:
            combo_kwargs = base_kwargs.copy()
            combo_kwargs.update(dict(zip(grid_keys, combo)))

            train_result = backtester.run(
                strategy_cls,
                data=train_data,
                capital=capital,
                strategy_kwargs=combo_kwargs,
            )
            metric_value, score = _evaluate_metric(train_result, metric)

            if score > best_score:
                best_score = score
                best_metric_value = metric_value
                best_params = combo_kwargs
                best_train_result = train_result

        if best_params is None or best_train_result is None:
            break

        test_result = backtester.run(
            strategy_cls,
            data=test_data,
            capital=capital,
            strategy_kwargs=best_params,
        )

        windows.append(
            WalkForwardWindowResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=dict(best_params),
                metric_value=best_metric_value if best_metric_value is not None else 0.0,
                train_result=best_train_result,
                test_result=test_result,
            )
        )

        start += step

    return WalkForwardResult(windows=windows)


def _slice_data(data_frames: Dict[str, pd.DataFrame], start: int, end: int) -> Optional[Dict[str, Frame]]:
    sliced: Dict[str, Frame] = {}
    for symbol, df in data_frames.items():
        window_df = df.iloc[start:end]
        if window_df.empty:
            return None
        sliced[symbol] = Frame.from_pandas(window_df.copy())
    return sliced


def _evaluate_metric(
    result: BacktestResult,
    metric: Union[str, Callable[[BacktestResult], float]],
) -> (float, float):
    if callable(metric):
        raw_value = float(metric(result))
    else:
        if not hasattr(result, metric):
            raise AttributeError(f"BacktestResult has no metric '{metric}'")
        attr = getattr(result, metric)
        raw_value = float(attr()) if callable(attr) else float(attr)

    if isinstance(metric, str) and metric == "max_drawdown":
        score = -abs(raw_value)
    else:
        score = raw_value

    return raw_value, score
