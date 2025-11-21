"""Factor library and pipeline API for alphaweave."""

from alphaweave.pipeline.factors import (
    Factor,
    ReturnsFactor,
    MomentumFactor,
    VolatilityFactor,
    BetaFactor,
    DollarVolumeFactor,
    SMAIndicatorFactor,
    RSIFactor,
)
from alphaweave.pipeline.filters import (
    Filter,
    TopN,
    BottomN,
    PercentileFilter,
    LiquidityFilter,
    VolatilityFilter,
    And,
    Or,
    Not,
)
from alphaweave.pipeline.expressions import FactorExpression
from alphaweave.pipeline.pipeline import Pipeline

# Sprint 13: Extended factors
try:
    from alphaweave.pipeline.factors_extended import (
        TimeSeriesMomentumFactor,
        CrossSectionalMomentumFactor,
        GarmanKlassVolFactor,
        ParkinsonVolFactor,
        ATRVolFactor,
        ReturnSkewFactor,
        ReturnKurtosisFactor,
        TrendSlopeFactor,
        TrendStrengthFactor,
        LowVolatilityFactor,
        TurnoverFactor,
        IdiosyncraticVolFactor,
        LogMarketCapFactor,
        BookToPriceFactor,
        EarningsToPriceFactor,
        DividendYieldFactor,
    )
    from alphaweave.pipeline.technical_factors import (
        MACDFactor,
        StochasticKFactor,
        BollingerZScoreFactor,
        CCIFactor,
        WilliamsRFactor,
    )
    from alphaweave.pipeline.regressions import RollingOLSRegressor, compute_factor_returns
    from alphaweave.pipeline.covariance import (
        shrink_cov_lw,
        shrink_cov_oas,
        compute_factor_covariances,
        condition_number,
        is_positive_semidefinite,
    )
    from alphaweave.pipeline.transforms import winsorize, normalize, lag, smooth

    _SPRINT_13_AVAILABLE = True
except ImportError:
    _SPRINT_13_AVAILABLE = False

__all__ = [
    # Factors (Sprint 12)
    "Factor",
    "ReturnsFactor",
    "MomentumFactor",
    "VolatilityFactor",
    "BetaFactor",
    "DollarVolumeFactor",
    "SMAIndicatorFactor",
    "RSIFactor",
    # Filters
    "Filter",
    "TopN",
    "BottomN",
    "PercentileFilter",
    "LiquidityFilter",
    "VolatilityFilter",
    "And",
    "Or",
    "Not",
    # Expressions
    "FactorExpression",
    # Pipeline
    "Pipeline",
]

if _SPRINT_13_AVAILABLE:
    __all__.extend([
        # Extended Factors (Sprint 13)
        "TimeSeriesMomentumFactor",
        "CrossSectionalMomentumFactor",
        "GarmanKlassVolFactor",
        "ParkinsonVolFactor",
        "ATRVolFactor",
        "ReturnSkewFactor",
        "ReturnKurtosisFactor",
        "TrendSlopeFactor",
        "TrendStrengthFactor",
        "LowVolatilityFactor",
        "TurnoverFactor",
        "IdiosyncraticVolFactor",
        "LogMarketCapFactor",
        "BookToPriceFactor",
        "EarningsToPriceFactor",
        "DividendYieldFactor",
        # Technical Factors
        "MACDFactor",
        "StochasticKFactor",
        "BollingerZScoreFactor",
        "CCIFactor",
        "WilliamsRFactor",
        # Regressions
        "RollingOLSRegressor",
        "compute_factor_returns",
        # Covariance
        "shrink_cov_lw",
        "shrink_cov_oas",
        "compute_factor_covariances",
        "condition_number",
        "is_positive_semidefinite",
        # Transforms
        "winsorize",
        "normalize",
        "lag",
        "smooth",
    ])

