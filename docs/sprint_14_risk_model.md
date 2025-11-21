# Sprint 14 — Barra-Lite Risk Model

**Status:** ✅ Completed

## Goal

Implement a multi-factor risk model that computes factor exposures, factor covariance, idiosyncratic variance, and risk decomposition, with integration into portfolio optimizers.

## What This Sprint Unlocks

- ✅ Factor-based portfolio optimization
- ✅ Hedged portfolios (factor-neutral)
- ✅ Style-neutral strategies
- ✅ Better weighting schemes using factor risk
- ✅ Risk-aware cross-sectional strategies
- ✅ Near-institutional research workflow

## Implementation Summary

### 1. Core Risk Model Classes

#### `RiskModel` (dataclass)
- `exposures`: Factor exposures matrix [symbols × factors]
- `factor_cov`: Factor covariance matrix [factors × factors]
- `specific_var`: Specific (idiosyncratic) variance per symbol [symbols]
- `total_covariance()`: Computes total asset covariance: B * F * B' + D
- `validate()`: Validates model consistency

#### `RiskDecomposition` (dataclass)
- `total_vol`: Total portfolio volatility (annualized)
- `factor_contrib`: Contribution of each factor to portfolio variance
- `specific_contrib`: Specific variance contribution per asset
- `marginal_risk`: Marginal risk contribution per asset
- `component_risk`: Component risk (w_i * marginal_risk_i) per asset

### 2. Factor Returns Estimation

#### `estimate_factor_returns()`
- Cross-sectional regression (Fama-MacBeth style)
- For each date: R_it = α_t + β_i1 f1_t + β_i2 f2_t + ... + ε_it
- Returns factor return series [date × factor]

### 3. Factor Covariance Estimation

#### `estimate_factor_covariance()`
- Methods:
  - `"sample"`: Sample covariance
  - `"ewma"`: Exponentially weighted moving average
  - `"shrinkage_lw"`: Ledoit-Wolf shrinkage (from Sprint 13)
  - `"shrinkage_oas"`: Oracle Approximating Shrinkage (from Sprint 13)
- Returns factor covariance matrix [factors × factors]

### 4. Specific Risk Estimation

#### `estimate_specific_risk()`
- Computes variance of regression residuals
- For each asset: ε_it = R_it - Σ_k β_ik * f_k,t
- Returns specific variance per symbol [symbol]

### 5. Exposures Computation

#### `compute_exposures()`
- Extracts factor exposures for a specific date
- Aligns factor data to common symbols
- Returns exposures matrix [symbol × factor]

#### `compute_exposures_rolling()`
- Computes rolling average exposures
- Useful for smoothing exposure estimates

### 6. Risk Decomposition

#### `decompose_risk()`
- Decomposes portfolio risk into factor and specific contributions
- Formula: Total Cov = B * F * B' + D
- Computes:
  - Total portfolio volatility
  - Factor risk contributions
  - Specific risk contributions
  - Marginal and component risk

### 7. Factor Hedging

#### `hedge_exposures()`
- Neutralizes portfolio exposure to specified factors
- Useful for beta-neutral, size-neutral, etc. portfolios
- Returns adjusted weights with neutralized exposures

### 8. Optimizer Integration

All optimizers now support optional `risk_model` parameter:
- `min_variance(risk_model=...)`: Uses risk model's total covariance
- `risk_parity(risk_model=...)`: Uses risk model's total covariance
- `target_volatility(risk_model=...)`: Uses risk model's total covariance

### 9. Files Created

- `alphaweave/portfolio/risk_model.py`: Core risk model implementation
- `examples/risk_model_basic.py`: Basic risk model usage
- `examples/factor_neutral_strategy.py`: Factor-neutral strategy
- `docs/sprint_14_risk_model.md`: This documentation

### 10. Files Modified

- `alphaweave/portfolio/optimizers.py`: Added `risk_model` parameter support
- `alphaweave/portfolio/__init__.py`: Export risk model components

## Usage Examples

### Basic Risk Model Construction

```python
from alphaweave.portfolio.risk_model import (
    RiskModel,
    estimate_factor_returns,
    estimate_factor_covariance,
    estimate_specific_risk,
    compute_exposures,
)

# Get factor data from pipeline
result = self.run_pipeline(pipeline, window="252D")
factor_data = result["factors"]

# Get asset returns
returns_df = self._get_returns(252)

# Build risk model
factor_returns = estimate_factor_returns(returns_df, factor_data)
factor_cov = estimate_factor_covariance(factor_returns, method="shrinkage_lw")
exposures = compute_exposures(factor_data, date=self.now())
specific_var = estimate_specific_risk(returns_df, factor_returns, exposures)

risk_model = RiskModel(
    exposures=exposures,
    factor_cov=factor_cov,
    specific_var=specific_var,
)
```

### Using Risk Model in Optimization

```python
from alphaweave.portfolio.optimizers import min_variance, risk_parity

# Use risk model instead of sample covariance
result = min_variance(risk_model=risk_model)
weights = result.weights

# Or with risk parity
result = risk_parity(risk_model=risk_model)
weights = result.weights
```

### Risk Decomposition

```python
from alphaweave.portfolio.risk_model import decompose_risk

# Decompose portfolio risk
risk_decomp = decompose_risk(weights, risk_model)

print(f"Total volatility: {risk_decomp.total_vol:.4f}")
print(f"Factor risk: {risk_decomp.factor_contrib.sum():.4f}")
print(f"Specific risk: {risk_decomp.specific_contrib.sum():.4f}")
print(f"Factor contributions:")
for factor, contrib in risk_decomp.factor_contrib.items():
    print(f"  {factor}: {contrib:.4f}")
```

### Factor-Neutral Portfolio

```python
from alphaweave.portfolio.risk_model import hedge_exposures

# Get initial weights
initial_weights = risk_parity(risk_model=risk_model).weights

# Neutralize beta exposure
adjusted_weights = hedge_exposures(
    initial_weights,
    risk_model.exposures,
    neutralize_factors=["beta"],
)
```

### Complete Strategy Example

```python
class MFPortfolio(Strategy):
    def init(self):
        self.pipeline = Pipeline(
            factors={
                "mom": MomentumFactor(window=63),
                "vol": VolatilityFactor(window=63),
                "beta": BetaFactor("SPY", window=252),
            },
        )
        self.lookback = 252

    def next(self, i):
        if not self.schedule.every("1M"):
            return

        # Get returns and factor data
        returns_df = self._get_returns(self.lookback)
        result = self.run_pipeline(self.pipeline, window=f"{self.lookback}D")
        factor_data = result["factors"]

        # Build risk model
        factor_returns = estimate_factor_returns(returns_df, factor_data)
        factor_cov = estimate_factor_covariance(factor_returns)
        exposures = compute_exposures(factor_data, date=self.now())
        specific_var = estimate_specific_risk(returns_df, factor_returns, exposures)

        risk_model = RiskModel(
            exposures=exposures,
            factor_cov=factor_cov,
            specific_var=specific_var,
        )

        # Optimize using risk model
        opt_result = risk_parity(risk_model=risk_model)
        weights = opt_result.weights

        # Apply weights
        for symbol, w in weights.items():
            self.order_target_percent(symbol, w)
```

## Design Notes

1. **Factor Returns Estimation**: Uses cross-sectional regression (Fama-MacBeth style) to estimate factor returns from asset returns and exposures.

2. **Total Covariance**: Risk model computes total covariance as B * F * B' + D, where:
   - B = exposures matrix
   - F = factor covariance
   - D = diagonal matrix of specific variances

3. **Optimizer Integration**: Optimizers accept either `cov_matrix` or `risk_model`. If `risk_model` is provided, uses its `total_covariance()` method.

4. **Factor Hedging**: `hedge_exposures()` adjusts weights to neutralize exposure to specified factors, useful for beta-neutral or style-neutral portfolios.

5. **Validation**: Risk model includes validation to ensure consistency of dimensions and alignment.

## Testing Status

⚠️ **Note:** Unit tests for Sprint 14 components are not yet implemented. Recommended test coverage:

- Risk model construction and validation
- Factor returns estimation accuracy
- Factor covariance estimation
- Specific risk estimation
- Risk decomposition correctness
- Factor hedging accuracy
- Optimizer integration with risk model

## Future Enhancements

- Additional factor models (Barra-style industry factors, etc.)
- Dynamic factor models (time-varying factor loadings)
- Factor attribution tools
- Risk budgeting with factor constraints
- Multi-period risk forecasting

## Conclusion

Sprint 14 successfully implements a Barra-lite multi-factor risk model that enables:
- Factor-based portfolio optimization
- Risk decomposition and attribution
- Factor-neutral portfolio construction
- Integration with existing optimizers
- Institutional-quality risk modeling workflow

The implementation provides a solid foundation for advanced portfolio construction and risk management, enabling strategies that can explicitly model and control factor exposures.

