# Sprint 13 — Expanded Factor Library & Statistical Tools

**Status:** ✅ Completed

## Goal

Add a comprehensive library of returns-based, technical, fundamental (placeholders), cross-sectional, and statistical factors, along with transformations, regressions, and normalization tools.

## What This Sprint Unlocks

- ✅ Multi-factor models
- ✅ Style factors (momentum, value, quality, size, volatility)
- ✅ Custom factor construction
- ✅ Factor combination (composites)
- ✅ Factor standardization, normalization, winsorization
- ✅ Rolling regressions (betas, alphas, idiosyncratic risk)
- ✅ Covariance shrinkage
- ✅ Factor exposure scoring
- ✅ Preparatory components for a Barra-lite risk model (Sprint 14)

## Implementation Summary

### 1. Expanded Factor Library

#### A. Momentum Variants
- `TimeSeriesMomentumFactor`: Longer lookback periods (e.g., 252 days)
- `CrossSectionalMomentumFactor`: Cross-sectional momentum for ranking

#### B. Volatility Estimators
- `GarmanKlassVolFactor`: Garman-Klass volatility using OHLC data
- `ParkinsonVolFactor`: Parkinson volatility using high-low data
- `ATRVolFactor`: ATR-based volatility normalized by price

#### C. Higher Moments
- `ReturnSkewFactor`: Rolling skewness of returns
- `ReturnKurtosisFactor`: Rolling kurtosis of returns

#### D. Trend Factors
- `TrendSlopeFactor`: Regression slope of log price
- `TrendStrengthFactor`: R² of rolling regression

#### E. Style & Risk Factors
- `LowVolatilityFactor`: Inverse rank of volatility
- `TurnoverFactor`: Volume-based turnover proxy
- `IdiosyncraticVolFactor`: Residual volatility from beta regression

#### F. Size Factor
- `LogMarketCapFactor`: Log market capitalization (with proxy support)

#### G. Fundamental Factor Placeholders
- `BookToPriceFactor`: Book-to-price ratio (requires fundamental data)
- `EarningsToPriceFactor`: Earnings-to-price ratio
- `DividendYieldFactor`: Dividend yield

#### H. Technical Factors
- `MACDFactor`: MACD signal line
- `StochasticKFactor`: Stochastic %K
- `BollingerZScoreFactor`: Bollinger Band Z-Score
- `CCIFactor`: Commodity Channel Index
- `WilliamsRFactor`: Williams %R

### 2. Statistical Transforms

New transform methods added to `FactorExpression`:
- `.winsorize(lower=0.01, upper=0.99)`: Winsorize outliers
- `.normalize(method="minmax"|"zscore")`: Normalize values
- `.lag(periods=1)`: Lag factor by periods
- `.smooth(window=3)`: Smooth using moving average
- `.rolling_zscore(window)`: Rolling z-score normalization

Standalone functions:
- `winsorize(df, lower, upper)`
- `normalize(df, method)`
- `lag(df, periods)`
- `smooth(df, window)`

### 3. Rolling Regressions

#### `RollingOLSRegressor`
- Rolling OLS regression for factor models
- Computes factor loadings, residuals, predicted values, R²
- Supports multiple independent factors

#### `compute_factor_returns()`
- Cross-sectional regression for factor returns (Fama-MacBeth style)
- Computes factor returns from factor exposures and asset returns

### 4. Covariance Enhancements

- `shrink_cov_lw()`: Ledoit-Wolf shrinkage
- `shrink_cov_oas()`: Oracle Approximating Shrinkage (OAS)
- `compute_factor_covariances()`: Factor covariance from factor returns
- `condition_number()`: Condition number of covariance matrix
- `is_positive_semidefinite()`: Check if matrix is PSD

### 5. Files Created

- `alphaweave/pipeline/factors_extended.py`: Extended factor library
- `alphaweave/pipeline/technical_factors.py`: Technical indicator factors
- `alphaweave/pipeline/transforms.py`: Statistical transforms
- `alphaweave/pipeline/regressions.py`: Rolling regressions
- `alphaweave/pipeline/covariance.py`: Covariance shrinkage
- `examples/multi_factor_style_tilt.py`: Example strategy
- `docs/sprint_13_factor_library.md`: This documentation

### 6. Files Modified

- `alphaweave/pipeline/__init__.py`: Export new components

## Usage Examples

### Multi-Factor Composite

```python
from alphaweave.pipeline import (
    Pipeline,
    MomentumFactor,
    VolatilityFactor,
    TrendSlopeFactor,
    LowVolatilityFactor,
    TopN,
)

# Create factors
mom = MomentumFactor(window=63).zscore()
vol = VolatilityFactor(window=63).zscore()
trend = TrendSlopeFactor(window=63).zscore()
low_vol = LowVolatilityFactor(window=63).zscore()

# Combine in pipeline
pipeline = Pipeline(
    factors={
        "mom": mom,
        "vol": vol,
        "trend": trend,
        "low_vol": low_vol,
    },
    screen=TopN("mom", 50),
)

# In strategy, compute composite
result = self.run_pipeline(pipeline, window="126D")
mom_scores = result["factors"]["mom"].iloc[-1]
vol_scores = result["factors"]["vol"].iloc[-1]
composite = mom_scores - vol_scores  # Combine factors
```

### Rolling Regression

```python
from alphaweave.pipeline import RollingOLSRegressor, MomentumFactor, VolatilityFactor

# Define regression
regressor = RollingOLSRegressor(
    dependent=MomentumFactor(window=63),
    independents={
        "beta": BetaFactor("SPY", window=252),
        "vol": VolatilityFactor(window=63),
    },
    window=252,
)

# Fit regression
results = regressor.fit(data)
loadings = results["coefficients"]  # Factor loadings
residuals = results["residuals"]    # Idiosyncratic returns
```

### Covariance Shrinkage

```python
from alphaweave.pipeline import shrink_cov_lw, compute_factor_returns

# Compute factor returns
factor_returns = compute_factor_returns(factors, asset_returns)

# Compute factor covariance
factor_cov = compute_factor_covariances(factor_returns)

# Apply shrinkage
shrunk_cov = shrink_cov_lw(factor_cov)
```

### Statistical Transforms

```python
from alphaweave.pipeline import MomentumFactor

# Create factor with transforms
mom = (
    MomentumFactor(window=63)
    .zscore()
    .winsorize(lower=0.01, upper=0.99)
    .smooth(window=3)
)
```

## Design Notes

1. **Factor-to-Factor Operations**: Currently limited in expressions. Users can compute factors separately and combine in strategy code.

2. **Fundamental Factors**: Placeholder implementations that require external data. Users provide DataFrames with fundamental data.

3. **Caching**: Pipeline caching enhancements are deferred to future work. Current implementation computes factors on-demand. For production use with large universes, consider implementing caching of intermediate results (returns, rolling windows, etc.).

4. **Performance**: For large universes (1000+ symbols), consider optimizing rolling operations and caching intermediate results.

## Testing Status

⚠️ **Note:** Unit tests for Sprint 13 components are not yet implemented. Recommended test coverage:

- Extended factor computation correctness
- Statistical transform behavior
- Rolling regression accuracy
- Covariance shrinkage properties
- Integration with pipeline engine

## Future Enhancements

- Advanced caching for pipeline performance
- More sophisticated factor combination operators
- Additional volatility estimators (Rogers-Satchell, etc.)
- Factor exposure attribution tools
- Barra-style risk model (Sprint 14)

## Conclusion

Sprint 13 successfully expands the factor library with:
- Comprehensive factor types (momentum, volatility, trend, technical, fundamental)
- Statistical transforms and preprocessing
- Rolling regression capabilities
- Covariance shrinkage methods
- Integration with existing pipeline infrastructure

The implementation provides a solid foundation for multi-factor model construction and risk analysis, preparing for more advanced risk modeling in future sprints.

