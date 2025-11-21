# Sprint 15 — Strategy Discovery & Auto-Alpha

**Status:** ✅ Completed

## Goal

Add tools that can search over strategy configurations and factor combinations, using automated backtesting, scoring, and overfitting controls.

## What This Sprint Unlocks

- ✅ Automated strategy parameter search
- ✅ Factor weight optimization
- ✅ Overfitting-aware evaluation
- ✅ Time-based cross-validation
- ✅ Parallel search execution
- ✅ Optional ML-based factor modeling

## Implementation Summary

### 1. Search Space Definition

#### `SearchSpace` (dataclass)
- `params`: Sequence of `Param` or `ContinuousParam`
- `grid_combinations()`: Generate all discrete parameter combinations
- `sample_random()`: Sample random configurations
- `size()`: Get search space size (None if continuous)

#### `Param` (dataclass)
- `name`: Parameter name
- `values`: List of discrete values

#### `ContinuousParam` (dataclass)
- `name`: Parameter name
- `low`: Lower bound
- `high`: Upper bound
- `log`: If True, use log-uniform sampling

### 2. Strategy Candidate Specification

#### `StrategyCandidateSpec` (dataclass)
- `name`: Candidate name
- `factory`: Function that creates Strategy class from parameters
- `search_space`: Search space definition
- `metadata`: Optional metadata

#### `StrategyFactory` (type alias)
- Callable that takes parameter dict and returns Strategy class

### 3. Evaluation Engine

#### `EvaluationConfig` (dataclass)
- `metric`: Metric to optimize ("sharpe", "cagr", "sortino", or callable)
- `min_trades`: Minimum trades required
- `time_splits`: Number of CV folds
- `use_walkforward`: Use walk-forward instead of time splits
- `overfit_penalty`: Apply overfitting penalty
- `stability_penalty`: Apply stability penalty

#### `StrategyEvalResult` (dataclass)
- `params`: Parameter configuration
- `score`: Composite score (higher is better)
- `metrics`: Dictionary of computed metrics
- `per_fold_scores`: Scores for each CV fold
- `backtest_results`: BacktestResult for each fold
- `diagnostics`: Optional diagnostic information

#### `evaluate_candidate()`
- Runs time-based cross-validation or walk-forward
- Computes metrics and applies penalties
- Returns StrategyEvalResult

### 4. Search Algorithms

#### `grid_search()`
- Exhaustively evaluates all discrete parameter combinations
- Supports parallel execution
- Returns sorted list of results

#### `random_search()`
- Samples random configurations from search space
- Supports continuous parameters
- Supports parallel execution
- Returns sorted list of results

### 5. ML Helpers (Optional)

#### `fit_factor_model()`
- Fits model to predict returns from factors
- Supports: "linear", "ridge", "random_forest"
- Requires scikit-learn (guarded import)

#### `model_to_scoring_function()`
- Converts fitted model to scoring function
- Can be used in generic signal strategies

### 6. Overfitting Controls

- **Time-based CV**: Splits data into train/test folds
- **Overfit penalty**: Penalizes gap between IS and OOS performance
- **Stability penalty**: Penalizes high variance across folds
- **Complexity penalty**: BIC-like penalty for parameter count

### 7. Files Created

- `alphaweave/alpha/__init__.py`: Module exports
- `alphaweave/alpha/search_space.py`: Search space definitions
- `alphaweave/alpha/candidates.py`: Strategy candidate specs
- `alphaweave/alpha/eval.py`: Evaluation engine
- `alphaweave/alpha/search.py`: Search algorithms
- `alphaweave/alpha/ml.py`: Optional ML helpers
- `examples/auto_sma_crossover_search.py`: SMA crossover search example
- `examples/auto_multifactor_weights_search.py`: Multi-factor weight search example
- `docs/sprint_15_auto_alpha.md`: This documentation

## Usage Examples

### Grid Search

```python
from alphaweave.alpha import (
    SearchSpace, Param, StrategyCandidateSpec,
    EvaluationConfig, grid_search,
)

# Define search space
space = SearchSpace(params=[
    Param("fast", values=[5, 10, 20]),
    Param("slow", values=[30, 50, 100]),
])

# Create candidate spec
spec = StrategyCandidateSpec(
    name="SMA_Cross",
    factory=sma_crossover_factory,
    search_space=space,
)

# Evaluation config
config = EvaluationConfig(
    metric="sharpe",
    time_splits=3,
    min_trades=20,
)

# Run search
results = grid_search(
    candidate=spec,
    data=data,
    eval_config=config,
    n_jobs=4,
)

# Best result
best = results[0]
print(f"Best params: {best.params}, Score: {best.score:.4f}")
```

### Random Search

```python
from alphaweave.alpha import (
    SearchSpace, ContinuousParam, StrategyCandidateSpec,
    EvaluationConfig, random_search,
)

# Define continuous search space
space = SearchSpace(params=[
    ContinuousParam("mom_weight", -1.0, 1.0),
    ContinuousParam("vol_weight", -1.0, 1.0),
])

# Run random search
results = random_search(
    candidate=spec,
    data=data,
    eval_config=config,
    n_samples=100,
    n_jobs=8,
    random_state=42,
)
```

### ML Factor Modeling

```python
from alphaweave.alpha.ml import fit_factor_model, model_to_scoring_function

# Fit model
model, diagnostics = fit_factor_model(
    factor_data=factor_df,
    future_returns=returns_series,
    model="ridge",
    alpha=1.0,
)

# Convert to scoring function
scoring_fn = model_to_scoring_function(model, factor_names=["mom", "vol"])

# Use in strategy
scores = scoring_fn(current_factor_df)
```

## Design Notes

1. **No Strategy API Changes**: Strategy discovery operates on parameterized Strategy classes, no changes to base Strategy API.

2. **Explicit Search Spaces**: Search spaces are explicit and declarative, not black-box AutoML.

3. **Overfitting Awareness**: Time-based CV, overfit penalties, and stability penalties help prevent overfitting.

4. **Optional ML**: ML helpers are optional and require scikit-learn. Graceful degradation if not installed.

5. **Parallel Execution**: Both grid and random search support parallel execution using ProcessPoolExecutor.

6. **Composability**: Search spaces, evaluation configs, and results can be combined and extended.

## Testing Status

⚠️ **Note:** Unit tests for Sprint 15 components are not yet implemented. Recommended test coverage:

- Search space generation (grid and random)
- Evaluation correctness
- Time-based CV splitting
- Overfitting penalty calculation
- Search algorithm correctness
- ML helper functionality (with sklearn)

## Future Enhancements

- Bayesian optimization (TPE, etc.)
- Multi-objective optimization
- Strategy ensemble methods
- Automated feature engineering
- More sophisticated overfitting controls

## Conclusion

Sprint 15 successfully implements strategy discovery and auto-alpha capabilities that enable:
- Automated parameter search
- Factor weight optimization
- Overfitting-aware evaluation
- Parallel search execution
- Optional ML-based modeling

The implementation provides a practical framework for systematic strategy exploration while maintaining awareness of overfitting risks through time-based CV and penalty mechanisms.

