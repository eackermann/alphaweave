"""Strategy evaluation and scoring for discovery."""

from dataclasses import dataclass
from typing import Mapping, Any, Callable, Optional, List
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.results.result import BacktestResult
from alphaweave.alpha.candidates import StrategyCandidateSpec


@dataclass
class EvaluationConfig:
    """
    Configuration for strategy evaluation.

    Attributes:
        metric: Metric to optimize ("sharpe", "cagr", "sortino", or callable)
        min_trades: Minimum number of trades required (default: 20)
        time_splits: Number of time-based CV folds (default: 3)
        use_walkforward: If True, use walk-forward instead of time splits (default: False)
        train_window: Training window size for walk-forward (required if use_walkforward=True)
        test_window: Test window size for walk-forward (required if use_walkforward=True)
        overfit_penalty: Apply overfitting penalty (default: True)
        stability_penalty: Apply stability penalty based on fold variance (default: True)
    """

    metric: str | Callable[[BacktestResult], float] = "sharpe"
    min_trades: int = 20
    time_splits: int = 3
    use_walkforward: bool = False
    train_window: Optional[int] = None
    test_window: Optional[int] = None
    overfit_penalty: bool = True
    stability_penalty: bool = True

    def __post_init__(self):
        """Validate evaluation config."""
        if self.use_walkforward:
            if self.train_window is None or self.test_window is None:
                raise ValueError(
                    "train_window and test_window required when use_walkforward=True"
                )


@dataclass
class StrategyEvalResult:
    """
    Result of evaluating a strategy candidate.

    Attributes:
        params: Parameter configuration that was evaluated
        score: Composite score (higher is better)
        metrics: Dictionary of computed metrics
        per_fold_scores: Scores for each CV fold
        backtest_results: BacktestResult for each fold
        diagnostics: Optional diagnostic information
    """

    params: Mapping[str, Any]
    score: float
    metrics: Mapping[str, float]
    per_fold_scores: List[float]
    backtest_results: List[BacktestResult]
    diagnostics: Optional[dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f"StrategyEvalResult(score={self.score:.4f}, "
            f"sharpe={self.metrics.get('sharpe', 'N/A'):.4f if isinstance(self.metrics.get('sharpe'), (int, float)) else 'N/A'}, "
            f"folds={len(self.backtest_results)})"
        )


def _extract_metric(result: BacktestResult, metric: str | Callable) -> float:
    """Extract metric value from BacktestResult."""
    if callable(metric):
        return metric(result)

    metric_lower = metric.lower()
    if metric_lower == "sharpe":
        return result.sharpe()
    elif metric_lower == "cagr":
        # Approximate CAGR from total return
        if result.total_return <= -1.0:
            return -1.0  # Cap at -100% return
        years = len(result.equity_series) / 252.0 if len(result.equity_series) > 0 else 1.0
        if years > 0:
            return (1.0 + result.total_return) ** (1.0 / years) - 1.0
        return result.total_return
    elif metric_lower == "sortino":
        # Sortino ratio (downside deviation)
        returns = result.returns
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return 0.0
        return (result.returns.mean() * 252) / downside_std
    elif metric_lower == "total_return":
        return result.total_return
    elif metric_lower == "max_drawdown":
        return -result.max_drawdown  # Negative because we want to maximize (less drawdown)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_overfit_penalty(
    in_sample_metric: float,
    out_of_sample_metric: float,
    n_params: int,
    n_observations: int,
) -> float:
    """
    Compute overfitting penalty.

    Penalty increases with:
    - Gap between IS and OOS performance
    - Number of parameters relative to observations (BIC-like)
    """
    # Gap penalty
    gap = in_sample_metric - out_of_sample_metric
    gap_penalty = max(0.0, gap) * 0.5  # Penalize positive gaps

    # Complexity penalty (BIC-like)
    if n_observations > 0:
        complexity_penalty = (n_params * np.log(n_observations)) / (2 * n_observations)
    else:
        complexity_penalty = 0.0

    return gap_penalty + complexity_penalty


def _compute_stability_penalty(fold_scores: List[float]) -> float:
    """
    Compute stability penalty based on variance across folds.

    Higher variance = less stable = higher penalty.
    """
    if len(fold_scores) < 2:
        return 0.0

    fold_scores_array = np.array(fold_scores)
    if np.std(fold_scores_array) == 0:
        return 0.0

    # Penalty proportional to coefficient of variation
    mean_score = np.mean(fold_scores_array)
    if abs(mean_score) < 1e-8:
        return 0.0

    cv = np.std(fold_scores_array) / abs(mean_score)
    return cv * 0.1  # Scale penalty


def _time_split_data(
    data: dict[str, Frame],
    n_splits: int,
) -> List[tuple[dict[str, Frame], dict[str, Frame]]]:
    """
    Split data into time-based train/test folds.

    Returns:
        List of (train_data, test_data) tuples
    """
    if not data:
        return []

    # Get common date range
    all_dates = set()
    for frame in data.values():
        df = frame.to_pandas()
        all_dates.update(df.index)

    if not all_dates:
        return []

    sorted_dates = sorted(all_dates)
    n_total = len(sorted_dates)

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    # Initialize lists
    train_dates_list = []
    test_dates_list = []

    if n_splits == 1:
        # Single split: use first 70% for train, last 30% for test
        split_idx = int(n_total * 0.7)
        train_dates = sorted_dates[:split_idx]
        test_dates = sorted_dates[split_idx:]
        if len(train_dates) > 0 and len(test_dates) > 0:
            train_dates_list.append(train_dates)
            test_dates_list.append(test_dates)
    else:
        # Multiple splits: divide into n_splits segments
        # Each fold uses all previous data for training, next segment for testing
        fold_size = n_total // (n_splits + 1)

        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_total)

            train_dates = sorted_dates[:train_end]
            test_dates = sorted_dates[test_start:test_end]

            if len(train_dates) > 0 and len(test_dates) > 0:
                train_dates_list.append(train_dates)
                test_dates_list.append(test_dates)

    # Create train/test data dictionaries
    splits = []
    for train_dates, test_dates in zip(train_dates_list, test_dates_list):
        train_data = {}
        test_data = {}

        for symbol, frame in data.items():
            df = frame.to_pandas()

            # Filter by dates
            train_df = df.loc[df.index.isin(train_dates)]
            test_df = df.loc[df.index.isin(test_dates)]

            if len(train_df) > 0:
                train_data[symbol] = Frame.from_pandas(train_df)
            if len(test_df) > 0:
                test_data[symbol] = Frame.from_pandas(test_df)

        if train_data and test_data:
            splits.append((train_data, test_data))

    return splits


def evaluate_candidate(
    candidate: StrategyCandidateSpec,
    params: Mapping[str, Any],
    data: dict[str, Frame],
    eval_config: EvaluationConfig,
    backtester_kwargs: Optional[Mapping[str, Any]] = None,
) -> StrategyEvalResult:
    """
    Evaluate a strategy candidate with given parameters.

    Args:
        candidate: Strategy candidate specification
        params: Parameter configuration to evaluate
        data: Data dictionary (symbol -> Frame)
        eval_config: Evaluation configuration
        backtester_kwargs: Optional backtester configuration

    Returns:
        StrategyEvalResult with scores and metrics
    """
    if backtester_kwargs is None:
        backtester_kwargs = {}

    # Create strategy class from factory
    strategy_cls = candidate.factory(params)

    # Prepare backtester
    backtester = VectorBacktester(**backtester_kwargs)
    capital = backtester_kwargs.get("capital", 100000.0)

    # Run evaluation based on config
    if eval_config.use_walkforward:
        # Walk-forward evaluation
        from alphaweave.analysis.walkforward import walk_forward_optimize

        # For walk-forward, we use the existing function but adapt it
        # For now, do simple time splits
        splits = _time_split_data(data, eval_config.time_splits)
        if not splits:
            # Fallback: single backtest
            result = backtester.run(strategy_cls, data=data, capital=capital)
            metric_value = _extract_metric(result, eval_config.metric)
            return StrategyEvalResult(
                params=params,
                score=metric_value,
                metrics={"sharpe": result.sharpe(), "total_return": result.total_return},
                per_fold_scores=[metric_value],
                backtest_results=[result],
            )
    else:
        # Time-based cross-validation
        splits = _time_split_data(data, eval_config.time_splits)

    if not splits:
        # Fallback: single backtest on all data
        result = backtester.run(strategy_cls, data=data, capital=capital)
        metric_value = _extract_metric(result, eval_config.metric)

        # Check min_trades
        if len(result.trades) < eval_config.min_trades:
            metric_value = -np.inf

        return StrategyEvalResult(
            params=params,
            score=metric_value,
            metrics={"sharpe": result.sharpe(), "total_return": result.total_return},
            per_fold_scores=[metric_value],
            backtest_results=[result],
        )

    # Run backtests on each fold
    fold_results = []
    fold_scores = []
    in_sample_metrics = []
    out_of_sample_metrics = []

    for train_data, test_data in splits:
        # Train on training data (optional: could optimize params here)
        train_result = backtester.run(strategy_cls, data=train_data, capital=capital)

        # Test on test data
        test_result = backtester.run(strategy_cls, data=test_data, capital=capital)

        # Check min_trades
        if len(test_result.trades) < eval_config.min_trades:
            fold_scores.append(-np.inf)
            fold_results.append(test_result)
            continue

        # Extract metrics
        train_metric = _extract_metric(train_result, eval_config.metric)
        test_metric = _extract_metric(test_result, eval_config.metric)

        in_sample_metrics.append(train_metric)
        out_of_sample_metrics.append(test_metric)
        fold_scores.append(test_metric)
        fold_results.append(test_result)

    if not fold_scores or all(s == -np.inf for s in fold_scores):
        # All folds failed min_trades
        return StrategyEvalResult(
            params=params,
            score=-np.inf,
            metrics={},
            per_fold_scores=fold_scores,
            backtest_results=fold_results,
        )

    # Compute aggregate metrics
    avg_oos_metric = np.mean([s for s in fold_scores if s != -np.inf])
    avg_is_metric = np.mean(in_sample_metrics) if in_sample_metrics else avg_oos_metric

    # Compute composite score
    score = avg_oos_metric
    overfit_penalty = 0.0
    stability_penalty = 0.0

    # Apply penalties
    if eval_config.overfit_penalty and len(in_sample_metrics) > 0:
        overfit_penalty = _compute_overfit_penalty(
            avg_is_metric,
            avg_oos_metric,
            n_params=len(params),
            n_observations=sum(len(r.equity_series) for r in fold_results),
        )
        score -= overfit_penalty

    if eval_config.stability_penalty:
        stability_penalty = _compute_stability_penalty(fold_scores)
        score -= stability_penalty

    # Aggregate metrics from all folds
    all_sharpes = [r.sharpe() for r in fold_results]
    all_returns = [r.total_return for r in fold_results]

    metrics = {
        "sharpe": np.mean(all_sharpes),
        "total_return": np.mean(all_returns),
        "sharpe_std": np.std(all_sharpes),
        "return_std": np.std(all_returns),
        "n_trades": sum(len(r.trades) for r in fold_results),
        "n_folds": len(fold_results),
    }

    return StrategyEvalResult(
        params=params,
        score=score,
        metrics=metrics,
        per_fold_scores=fold_scores,
        backtest_results=fold_results,
        diagnostics={
            "in_sample_metric": avg_is_metric,
            "out_of_sample_metric": avg_oos_metric,
            "overfit_penalty": overfit_penalty if eval_config.overfit_penalty else 0.0,
            "stability_penalty": stability_penalty if eval_config.stability_penalty else 0.0,
        },
    )

