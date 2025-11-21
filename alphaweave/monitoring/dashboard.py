"""HTML dashboard generation utilities."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from alphaweave.analysis.drift import compute_live_drift_series
from alphaweave.monitoring import plots
from alphaweave.monitoring.run import RunMonitor


def _fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _series_to_metrics(series: pd.Series) -> dict:
    if series.empty:
        return {"final_equity": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
    final_eq = float(series.iloc[-1])
    start_eq = float(series.iloc[0]) if series.iloc[0] != 0 else 1.0
    total_return = (final_eq / start_eq) - 1.0 if start_eq else 0.0
    drawdown = float((series / series.cummax() - 1.0).min())
    return {
        "final_equity": final_eq,
        "total_return": total_return,
        "max_drawdown": drawdown,
    }


def generate_html_dashboard(
    run: RunMonitor,
    title: str = "Alphaweave Run Dashboard",
    *,
    backtest_equity: Optional[pd.Series] = None,
) -> str:
    """Produce a simple HTML dashboard summarizing a run."""
    sections = [f"<h1>{title}</h1>"]

    equity = run.equity_curve()
    metrics = _series_to_metrics(equity)
    sections.append(
        "<h2>Overview</h2>"
        "<ul>"
        f"<li>Final equity: {metrics['final_equity']:.2f}</li>"
        f"<li>Total return: {metrics['total_return']:.2%}</li>"
        f"<li>Max drawdown: {metrics['max_drawdown']:.2%}</li>"
        "</ul>"
    )

    # Equity & drawdown
    eq_fig = plots.plot_equity_and_drawdown(run)
    sections.append("<h2>Equity & Drawdown</h2>")
    sections.append(f"<img src='data:image/png;base64,{_fig_to_base64(eq_fig)}' />")

    # Exposure heatmap
    sections.append("<h2>Exposure</h2>")
    exp_fig = plots.plot_exposure_heatmap(run)
    sections.append(f"<img src='data:image/png;base64,{_fig_to_base64(exp_fig)}' />")

    # Turnover
    sections.append("<h2>Turnover</h2>")
    turnover_fig = plots.plot_turnover(run)
    sections.append(f"<img src='data:image/png;base64,{_fig_to_base64(turnover_fig)}' />")

    # Trade PnL
    sections.append("<h2>Trade PnL Distribution</h2>")
    trade_fig = plots.plot_trade_pnl_histogram(run)
    sections.append(f"<img src='data:image/png;base64,{_fig_to_base64(trade_fig)}' />")

    # Trades preview
    trades = run.trades.head(20)
    if not trades.empty:
        trades_html = trades.to_html(index=False)
        sections.append("<h2>Recent Trades</h2>")
        sections.append(trades_html)

    # Drift comparison if applicable
    if backtest_equity is not None and not backtest_equity.empty:
        drift = compute_live_drift_series(backtest_equity, run)
        if not drift.empty:
            sections.append("<h2>Live vs Backtest Drift</h2>")
            drift_fig = _plot_drift(backtest_equity, run, drift)
            sections.append(f"<img src='data:image/png;base64,{_fig_to_base64(drift_fig)}' />")
            sections.append(
                f"<p>Current drift: {drift.iloc[-1]:.2%}, "
                f"Max drift: {drift.min():.2%} / {drift.max():.2%}</p>"
            )

    html = "<html><head><meta charset='utf-8'></head><body>{}</body></html>".format(
        "".join(sections)
    )
    return html


def _plot_drift(backtest_equity: pd.Series, run: RunMonitor, drift: pd.Series):
    backtest = backtest_equity.sort_index()
    live = run.equity_curve()
    aligned_backtest = backtest.reindex(live.index, method="ffill").dropna()
    aligned_live = live.loc[aligned_backtest.index]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(aligned_backtest.index, aligned_backtest.values, label="Backtest", color="tab:blue")
    ax1.plot(aligned_live.index, aligned_live.values, label="Live", color="tab:orange")
    ax1.set_ylabel("Equity")
    ax1.legend()
    ax1.set_title("Live vs Backtest Equity")

    ax2.plot(drift.index, drift.values, color="tab:green")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Drift")
    ax2.set_xlabel("Date")
    ax2.set_title("Equity Drift (pct)")
    fig.tight_layout()
    return fig


