"""Portfolio construction and optimization tools."""

from alphaweave.portfolio.constraints import PortfolioConstraints, WeightBounds
from alphaweave.portfolio.costs import (
    CompositeCostModel,
    ImpactCostModel,
    ProportionalCostModel,
    SpreadBasedCostModel,
    TransactionCostModel,
)
from alphaweave.portfolio.optimizers import (
    OptimizationResult,
    equal_weight,
    mean_variance,
    min_variance,
    risk_parity,
    target_volatility,
)
from alphaweave.portfolio.risk import estimate_covariance, estimate_volatility
from alphaweave.portfolio.turnover import (
    RebalancePenalty,
    TurnoverConstraint,
    compute_turnover,
)
from alphaweave.portfolio.universe import normalize_scores_to_weights, top_n_by_score

# Sprint 14: Risk model
try:
    from alphaweave.portfolio.risk_model import (
        RiskModel,
        RiskDecomposition,
        estimate_factor_returns,
        estimate_factor_covariance,
        estimate_specific_risk,
        compute_exposures,
        compute_exposures_rolling,
        decompose_risk,
        hedge_exposures,
    )
    _RISK_MODEL_AVAILABLE = True
except ImportError:
    _RISK_MODEL_AVAILABLE = False

__all__ = [
    "OptimizationResult",
    "PortfolioConstraints",
    "WeightBounds",
    "TransactionCostModel",
    "ProportionalCostModel",
    "SpreadBasedCostModel",
    "ImpactCostModel",
    "CompositeCostModel",
    "equal_weight",
    "mean_variance",
    "min_variance",
    "risk_parity",
    "target_volatility",
    "estimate_covariance",
    "estimate_volatility",
    "top_n_by_score",
    "normalize_scores_to_weights",
    "compute_turnover",
    "TurnoverConstraint",
    "RebalancePenalty",
]

if _RISK_MODEL_AVAILABLE:
    __all__.extend([
        "RiskModel",
        "RiskDecomposition",
        "estimate_factor_returns",
        "estimate_factor_covariance",
        "estimate_specific_risk",
        "compute_exposures",
        "compute_exposures_rolling",
        "decompose_risk",
        "hedge_exposures",
    ])

