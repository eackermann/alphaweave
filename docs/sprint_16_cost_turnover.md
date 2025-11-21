# Sprint 16 — Cost-aware & Turnover-aware Optimization

**Status:** 🚧 In progress

## Goal

Extend the portfolio layer so that optimizers and strategies can explicitly trade off performance versus realistic trading frictions. Transaction costs, market impact, and turnover constraints now plug directly into `alphaweave.portfolio`.

## Key Outcomes

- Portfolio-level transaction cost models (`Proportional`, `SpreadBased`, `Impact`, `Composite`).
- Turnover helpers: `compute_turnover`, `TurnoverConstraint`, `RebalancePenalty`.
- Optimizers (`mean_variance`, `min_variance`, `risk_parity`) accept `prev_weights`, cost models, turnover caps, and soft penalties.
- Backtest results expose realized cost diagnostics for calibration.
- Examples + docs showing the strategy pattern (track `prev_weights`, pass them into optimizers, keep Strategy API unchanged).

## 1. Transaction Cost Models (`alphaweave/portfolio/costs.py`)

```12:52:alphaweave/portfolio/costs.py
@dataclass
class ProportionalCostModel(TransactionCostModel):
    cost_per_dollar: pd.Series | float

    def estimate_cost(...):
        ...
```

- `TransactionCostModel`: base class returning fractional cost relative to portfolio value.
- `ProportionalCostModel`: linear turnover * cost_per_dollar (series or scalar).
- `SpreadBasedCostModel`: approximates ½ spread * participation * turnover.
- `ImpactCostModel`: square-root impact using ADV scaling.
- `CompositeCostModel`: sum of multiple components.

Synthetic tests ensure scaling behaves as expected.

## 2. Turnover Helpers (`alphaweave/portfolio/turnover.py`)

- `compute_turnover(prev, target)`: sum of absolute weight changes.
- `TurnoverConstraint`: `max_turnover` + `max_change_per_asset`.
- `RebalancePenalty`: λ for soft shrinkage back to previous weights.
- `apply_turnover_constraints` + `apply_rebalance_penalty`: heuristics that scale/clamp deltas without changing Strategy semantics.

## 3. Optimizer Extensions (`alphaweave/portfolio/optimizers.py`)

All Sprint 11 optimizers now accept:

```python
prev_weights: pd.Series | None = None
transaction_cost_model: TransactionCostModel | None = None
turnover_constraint: TurnoverConstraint | None = None
rebalance_penalty: RebalancePenalty | None = None
prices: pd.Series | None = None
volumes: pd.Series | None = None
```

- Cost models enter the optimization objective (e.g., Sharpe – cost).
- After solving, heuristics enforce turnover caps and soft penalties.
- Diagnostics include estimated cost + realized turnover so strategies can log/monitor.

## 4. BacktestResult Hooks (`alphaweave/results/result.py`)

- `realized_cost_series()`: cumulative fees + slippage.
- `realized_cost_per_turnover()`: average cost per unit turnover based on historical trades.
- Use these metrics to calibrate cost models (`lambda_cost ≈ realized_cost / turnover`).

## 5. Examples

- `examples/cost_aware_rebalance_equal_weight.py`: monthly rebalance with turnover cap.
- `examples/cost_aware_multifactor_portfolio.py`: composite cost model + rebalance penalty.
- `examples/compare_cost_aware_vs_naive.py`: runs both naive and cost-aware versions to highlight differences in realized costs.

## 6. API / Docs Updates

- `API.md` documents new cost + turnover modules and optimizer arguments.
- README snippets reference the cost-aware strategy pattern.
- This sprint doc (Sprint 16) explains the mental model: Strategy tracks `prev_weights`, passes them into optimizers, and realism is opt-in so legacy code keeps working.


