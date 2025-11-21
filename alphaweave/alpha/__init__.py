"""Strategy discovery and auto-alpha tools."""

from alphaweave.alpha.search_space import SearchSpace, Param, ContinuousParam
from alphaweave.alpha.candidates import StrategyCandidateSpec, StrategyFactory
from alphaweave.alpha.eval import (
    EvaluationConfig,
    StrategyEvalResult,
    evaluate_candidate,
)
from alphaweave.alpha.search import grid_search, random_search

# Optional ML helpers
try:
    from alphaweave.alpha.ml import fit_factor_model
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

__all__ = [
    "SearchSpace",
    "Param",
    "ContinuousParam",
    "StrategyCandidateSpec",
    "StrategyFactory",
    "EvaluationConfig",
    "StrategyEvalResult",
    "evaluate_candidate",
    "grid_search",
    "random_search",
]

if _ML_AVAILABLE:
    __all__.append("fit_factor_model")

