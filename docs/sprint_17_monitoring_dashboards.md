# Sprint 17 — Monitoring, Logging & Dashboards

**Status:** 🚧 In progress

## Goal

Introduce a monitoring layer that works for both backtests and (future) live runs. Engines now emit structured events that are easy to inspect, plot, and embed inside lightweight dashboards.

## Key Outcomes

- New `alphaweave.monitoring` package
  - `BarSnapshot`, `TradeRecord`, and `Monitor` protocol
  - `InMemoryMonitor` that stores events and exposes pandas DataFrames
  - `RunMonitor` helper with derived metrics (equity, drawdown, exposure, turnover, costs)
- Engines
  - `VectorBacktester.run(..., monitor=...)` wires monitoring hooks
  - Minimal `LiveEngine` wrapper shares the same monitoring API
  - Strategies get `log_metric()` to publish custom scalar telemetry
- Visualization
  - `monitoring.plots` exposes backend-agnostic Matplotlib helpers
  - `monitoring.dashboard.generate_html_dashboard()` builds a reusable HTML report
- Drift analysis
  - `analysis.compute_live_drift_series` compares live monitoring vs backtest equity
- Docs + examples:
  - `examples/backtest_with_dashboard.py`
  - `examples/live_replay_with_dashboard.py`
  - `examples/strategy_custom_metrics_logging.py`

## Monitoring Core (`alphaweave/monitoring/core.py`)

```python
class Monitor(Protocol):
    def on_run_start(self, meta: Mapping[str, Any] | None = None) -> None: ...
    def on_bar(self, snapshot: BarSnapshot) -> None: ...
    def on_trade(self, trade: TradeRecord) -> None: ...
    def on_metric(self, name: str, value: float, timestamp: datetime) -> None: ...
    def on_run_end(self) -> None: ...
```

`InMemoryMonitor` stores bars, trades, and metrics in lists and exposes `bars_df()`, `trades_df()`, `metrics_df()` for notebook-friendly inspection.

## RunMonitor (`alphaweave/monitoring/run.py`)

Wraps an `InMemoryMonitor` and provides derived analytics:

- `equity_curve()`, `drawdown_curve()`
- `exposure_over_time()` weights per symbol
- `turnover_over_time()` simple 0.5 * sum(|Δw|)
- `cost_over_time()` cumulative fees + slippage from trades

## Engine Integration

- `VectorBacktester.run(..., monitor=None, monitor_meta=None)`
  - Emits `on_run_start`, per-bar `BarSnapshot`, per-fill `TradeRecord`, and `on_run_end`
  - Supplies metadata (`mode`, `performance_mode`, etc.) for dashboards
- Strategies automatically receive `_monitor` and can call `self.log_metric(name, value)`
- `LiveEngine` (new package) reuses `VectorBacktester` but tags runs with `mode="live"`

## Visualization & Dashboards

- `monitoring.plots` includes `plot_equity_and_drawdown`, `plot_exposure_heatmap`, `plot_turnover`, `plot_trade_pnl_histogram`
- `monitoring.dashboard.generate_html_dashboard(...)`
  - Embeds Matplotlib PNGs as base64
  - Shows overview metrics, exposures, turnover, trades table, and optional backtest vs live drift comparison

## Drift Monitoring (`alphaweave/analysis/drift.py`)

`compute_live_drift_series(backtest_equity, live_run)` aligns backtest equity with live monitoring data and returns `(live / backtest) - 1`. Dashboards surface this when backtest equity is supplied.

## Examples

- `backtest_with_dashboard.py` — run a multi-asset backtest, emit monitor events, and write an HTML dashboard
- `live_replay_with_dashboard.py` — reuse the monitoring stack with the new `LiveEngine`
- `strategy_custom_metrics_logging.py` — demonstrate `Strategy.log_metric` piping into dashboards

## Next Steps

- Extend `LiveEngine` with real broker/datafeed integrations
- Add Plotly/Streamlit front-ends leveraging the same `RunMonitor`
- Stream monitor events over sockets for distributed deployments


