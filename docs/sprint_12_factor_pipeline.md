# Sprint 12 — Factor Library & Pipeline API

**Status:** ✅ Completed

## Goal

Introduce a factor computation system and a declarative pipeline engine for computing cross-sectional and time-series factors, performing universe screening, and driving portfolio optimization.

## What This Sprint Unlocks

- ✅ Cross-sectional factor strategies
- ✅ Multi-factor models
- ✅ Universe filtering (liquidity, volatility, momentum screens)
- ✅ Lightweight risk model inputs (betas, idiosyncratic vol)
- ✅ Efficient batch computation of rolling indicators/signals across many symbols
- ✅ All while preserving Strategy API simplicity

## Implementation Summary

### 1. New Modules Created

#### `alphaweave/pipeline/factors.py`
- `Factor` base class
- Built-in factors:
  - `ReturnsFactor`: Returns over specified window
  - `MomentumFactor`: Lookback return (momentum)
  - `VolatilityFactor`: Realized volatility (rolling std of returns)
  - `BetaFactor`: Rolling beta to benchmark
  - `DollarVolumeFactor`: Average dollar volume
  - `SMAIndicatorFactor`: SMA ratio (price / SMA)
  - `RSIFactor`: Relative Strength Index

#### `alphaweave/pipeline/filters.py`
- `Filter` base class
- Built-in filters:
  - `TopN`: Select top N symbols by factor
  - `BottomN`: Select bottom N symbols by factor
  - `PercentileFilter`: Filter by percentile threshold
  - `LiquidityFilter`: Filter by dollar volume
  - `VolatilityFilter`: Filter by volatility
  - `And`, `Or`, `Not`: Combinators for filters

#### `alphaweave/pipeline/expressions.py`
- `FactorExpression` class with:
  - Operator overloads: `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`
  - Transformations: `.zscore()`, `.rank()`, `.percentile()`, `.mean()`, `.std()`

#### `alphaweave/pipeline/pipeline.py`
- `Pipeline` class with:
  - `factors`: Dictionary of factor definitions
  - `filters`: Dictionary of filter definitions
  - `screen`: Final screen filter
  - `run()`: Execute pipeline and return results

### 2. Strategy Integration

Added `Strategy.run_pipeline()` method that:
- Accepts a `Pipeline` instance
- Optionally accepts a `window` parameter (e.g., "3M", "126D")
- Returns a dictionary with:
  - `"factors"`: Dictionary of factor name -> DataFrame
  - `"filters"`: Dictionary of filter name -> DataFrame (boolean masks)
  - `"screen"`: Final screen DataFrame (boolean mask)

### 3. Examples Created

1. **`examples/cross_sectional_momentum_pipeline.py`**: Momentum strategy using pipeline
2. **`examples/low_volatility_portfolio.py`**: Low volatility portfolio using pipeline
3. **`examples/multi_factor_composite.py`**: Multi-factor composite strategy
4. **`examples/pipeline_with_optimizer.py`**: Pipeline integrated with portfolio optimizers

### 4. Documentation

- Added "Factor Library & Pipeline API" section to `API.md`
- Created this sprint documentation file

## Design Principles Achieved

✅ **Factor computation is:**
- Vectorized (fast for large universes)
- Lazy, cached, and reusable
- Date-aligned and robust

✅ **Pipeline is:**
- Declarative: users specify what they want, not how
- Composable: factors, filters, masks, rankings can be mixed
- Time-indexed: pipeline outputs a panel of values [date × symbol]
- Pluggable into strategies

✅ **Strategy API preserved:**
- No modifications to existing Strategy API
- New `run_pipeline()` helper method added
- Backward compatible

## Usage Examples

### Basic Pipeline

```python
from alphaweave.pipeline import Pipeline, MomentumFactor, TopN

pipeline = Pipeline(
    factors={"mom": MomentumFactor(window=63)},
    screen=TopN("mom", 50),
)

result = pipeline.run(data)
mom_scores = result["factors"]["mom"]
screen_mask = result["screen"]
```

### Multi-Factor Composite

```python
from alphaweave.pipeline import (
    Pipeline, MomentumFactor, VolatilityFactor, RSIFactor, TopN
)

mom = MomentumFactor(window=63).zscore()
vol = VolatilityFactor(window=63).zscore()
rsi = RSIFactor(period=14).zscore()

composite = mom - vol + (rsi - 50) / 50

pipeline = Pipeline(
    factors={
        "mom": mom,
        "vol": vol,
        "rsi": rsi,
        "composite": composite,
    },
    screen=TopN("composite", 50),
)
```

### Strategy Integration

```python
class MultiFactorStrategy(Strategy):
    def init(self):
        self.pipeline = Pipeline(
            factors={"mom": MomentumFactor(window=63)},
            screen=TopN("mom", 50),
        )

    def next(self, i):
        if not self.schedule.every("1M"):
            return

        result = self.run_pipeline(self.pipeline, window="126D")
        mom_scores = result["factors"]["mom"].iloc[-1]
        screen = result["screen"].iloc[-1]

        selected = [s for s in screen.index if screen[s]]
        for symbol in selected:
            self.order_target_percent(symbol, 1.0 / len(selected))
```

## Testing Status

⚠️ **Note:** Unit tests for the pipeline system are not yet implemented. Recommended test coverage:

- Factor computation correctness
- Filter logic correctness
- Expression transformations
- Pipeline execution
- Strategy integration
- Multi-symbol alignment
- Edge cases (missing data, single symbol, etc.)

## Future Enhancements (Not in This Sprint)

- Multi-threading or distributed runs
- CVX or heavy dependencies for advanced optimizers
- Intraday pipeline (Sprint 6 intraday engine stays independent)
- More sophisticated caching strategies
- Factor-to-factor operations in expressions (currently handled by pipeline engine)

## Files Changed/Created

### New Files
- `alphaweave/pipeline/__init__.py`
- `alphaweave/pipeline/factors.py`
- `alphaweave/pipeline/filters.py`
- `alphaweave/pipeline/expressions.py`
- `alphaweave/pipeline/pipeline.py`
- `examples/cross_sectional_momentum_pipeline.py`
- `examples/low_volatility_portfolio.py`
- `examples/multi_factor_composite.py`
- `examples/pipeline_with_optimizer.py`
- `docs/sprint_12_factor_pipeline.md`

### Modified Files
- `alphaweave/strategy/base.py`: Added `run_pipeline()` method
- `API.md`: Added "Factor Library & Pipeline API" section

## Conclusion

Sprint 12 successfully introduces a comprehensive factor library and pipeline API that enables:
- Declarative factor computation
- Flexible universe screening
- Multi-factor model construction
- Seamless integration with existing Strategy API

The implementation follows the design principles and maintains backward compatibility while providing powerful new capabilities for cross-sectional and multi-factor strategies.

