"""Search algorithms for strategy discovery."""

from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
from alphaweave.alpha.candidates import StrategyCandidateSpec
from alphaweave.alpha.eval import EvaluationConfig, StrategyEvalResult, evaluate_candidate
from alphaweave.core.frame import Frame


def _evaluate_single_config(args: tuple) -> StrategyEvalResult:
    """
    Worker function for parallel evaluation.

    Must be top-level for pickling.
    """
    candidate, params, data, eval_config, backtester_kwargs = args
    return evaluate_candidate(
        candidate=candidate,
        params=params,
        data=data,
        eval_config=eval_config,
        backtester_kwargs=backtester_kwargs,
    )


def grid_search(
    candidate: StrategyCandidateSpec,
    data: dict[str, Frame],
    eval_config: EvaluationConfig,
    backtester_kwargs: Optional[dict] = None,
    n_jobs: int = 1,
) -> List[StrategyEvalResult]:
    """
    Exhaustively evaluate all combinations of discrete parameters.

    Args:
        candidate: Strategy candidate specification
        data: Data dictionary (symbol -> Frame)
        eval_config: Evaluation configuration
        backtester_kwargs: Optional backtester configuration
        n_jobs: Number of parallel jobs (1 = sequential, >1 = parallel)

    Returns:
        List of StrategyEvalResult sorted by score (highest first)
    """
    if backtester_kwargs is None:
        backtester_kwargs = {}

    # Get all parameter combinations
    param_combos = list(candidate.search_space.grid_combinations())

    if not param_combos:
        return []

    # Prepare jobs
    jobs = [
        (candidate, params, data, eval_config, backtester_kwargs)
        for params in param_combos
    ]

    # Execute jobs
    if n_jobs == 1:
        # Sequential execution
        results = []
        for job in jobs:
            result = _evaluate_single_config(job)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_evaluate_single_config, jobs))

    # Sort by score (highest first)
    results.sort(key=lambda r: r.score, reverse=True)

    return results


def random_search(
    candidate: StrategyCandidateSpec,
    data: dict[str, Frame],
    eval_config: EvaluationConfig,
    n_samples: int = 50,
    backtester_kwargs: Optional[dict] = None,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
) -> List[StrategyEvalResult]:
    """
    Sample random configurations from the search space.

    Args:
        candidate: Strategy candidate specification
        data: Data dictionary (symbol -> Frame)
        eval_config: Evaluation configuration
        n_samples: Number of random samples to evaluate
        backtester_kwargs: Optional backtester configuration
        n_jobs: Number of parallel jobs (1 = sequential, >1 = parallel)
        random_state: Random seed for reproducibility

    Returns:
        List of StrategyEvalResult sorted by score (highest first)
    """
    if backtester_kwargs is None:
        backtester_kwargs = {}

    # Sample random parameter configurations
    param_combos = candidate.search_space.sample_random(n_samples, random_state=random_state)

    if not param_combos:
        return []

    # Prepare jobs
    jobs = [
        (candidate, params, data, eval_config, backtester_kwargs)
        for params in param_combos
    ]

    # Execute jobs
    if n_jobs == 1:
        # Sequential execution
        results = []
        for job in jobs:
            result = _evaluate_single_config(job)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_evaluate_single_config, jobs))

    # Sort by score (highest first)
    results.sort(key=lambda r: r.score, reverse=True)

    return results

