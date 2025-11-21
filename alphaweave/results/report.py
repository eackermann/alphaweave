"""Report generation for backtest results."""

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphaweave.results.result import BacktestResult


def generate_markdown_report(
    result: BacktestResult,
    *,
    title: str = "Backtest Report",
    benchmark: Optional[pd.Series] = None,
    factor_returns: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a Markdown report summarizing the backtest.

    Args:
        result: BacktestResult to report on
        title: Report title
        benchmark: Optional benchmark returns for comparison
        factor_returns: Optional factor returns for regression

    Returns:
        Markdown string
    """
    lines = [f"# {title}", ""]

    # Overview
    lines.append("## Overview")
    lines.append("")
    if isinstance(result.equity_series.index, pd.DatetimeIndex):
        start_date = result.equity_series.index[0]
        end_date = result.equity_series.index[-1]
        lines.append(f"- **Start Date:** {start_date}")
        lines.append(f"- **End Date:** {end_date}")
    lines.append(f"- **Number of Bars:** {len(result.equity_series)}")
    lines.append(f"- **Number of Trades:** {len(result.trades)}")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append(f"- **Total Return:** {result.total_return:.2%}")
    lines.append(f"- **Final Equity:** ${result.final_equity:,.2f}")
    lines.append(f"- **Sharpe Ratio:** {result.sharpe():.2f}")
    lines.append(f"- **Max Drawdown:** {result.max_drawdown:.2%}")
    lines.append("")

    # Trade Summary
    lines.append("## Trade Summary")
    lines.append("")
    trade_summary = result.trade_summary()
    lines.append(f"- **Number of Trades:** {trade_summary['n_trades']}")
    lines.append(f"- **Win Rate:** {trade_summary['win_rate']:.2%}")
    lines.append(f"- **Average Win:** ${trade_summary['avg_win']:.2f}")
    lines.append(f"- **Average Loss:** ${trade_summary['avg_loss']:.2f}")
    lines.append(f"- **Expectancy:** ${trade_summary['expectancy']:.2f}")
    lines.append(f"- **Max Consecutive Wins:** {trade_summary['max_consecutive_wins']}")
    lines.append(f"- **Max Consecutive Losses:** {trade_summary['max_consecutive_losses']}")
    lines.append("")

    # Turnover & Costs
    lines.append("## Turnover & Costs")
    lines.append("")
    try:
        turnover = result.turnover("1M")
        if not turnover.empty:
            avg_turnover = turnover.mean()
            lines.append(f"- **Average Monthly Turnover:** {avg_turnover:.2%}")
    except Exception:
        pass

    avg_slippage = result.average_slippage_per_share()
    if avg_slippage > 0:
        lines.append(f"- **Average Slippage per Share:** ${avg_slippage:.4f}")
    lines.append("")

    # Factor Regression (if provided)
    if factor_returns is not None:
        lines.append("## Factor Regression")
        lines.append("")
        try:
            factor_result = result.factor_regression(factor_returns)
            lines.append(f"- **Alpha:** {factor_result.alpha:.4f}")
            lines.append(f"- **R²:** {factor_result.r2:.4f}")
            lines.append(f"- **Number of Observations:** {factor_result.n_obs}")
            lines.append("")
            lines.append("**Betas:**")
            for factor, beta in factor_result.betas.items():
                lines.append(f"- {factor}: {beta:.4f}")
        except Exception as e:
            lines.append(f"*Factor regression failed: {e}*")
        lines.append("")

    return "\n".join(lines)


def _plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def generate_html_report(
    result: BacktestResult,
    *,
    title: str = "Backtest Report",
    benchmark: Optional[pd.Series] = None,
    factor_returns: Optional[pd.DataFrame] = None,
    include_plots: bool = True,
) -> str:
    """
    Generate an HTML report with optional embedded plots.

    Args:
        result: BacktestResult to report on
        title: Report title
        benchmark: Optional benchmark returns for comparison
        factor_returns: Optional factor returns for regression
        include_plots: If True, embed matplotlib plots as base64 images

    Returns:
        HTML string
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #666; margin-top: 30px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "img { max-width: 100%; height: auto; margin: 20px 0; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]

    # Overview
    html_parts.append("<h2>Overview</h2>")
    html_parts.append("<ul>")
    if isinstance(result.equity_series.index, pd.DatetimeIndex):
        start_date = result.equity_series.index[0]
        end_date = result.equity_series.index[-1]
        html_parts.append(f"<li><strong>Start Date:</strong> {start_date}</li>")
        html_parts.append(f"<li><strong>End Date:</strong> {end_date}</li>")
    html_parts.append(f"<li><strong>Number of Bars:</strong> {len(result.equity_series)}</li>")
    html_parts.append(f"<li><strong>Number of Trades:</strong> {len(result.trades)}</li>")
    html_parts.append("</ul>")

    # Performance Metrics
    html_parts.append("<h2>Performance Metrics</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Total Return:</strong> {result.total_return:.2%}</li>")
    html_parts.append(f"<li><strong>Final Equity:</strong> ${result.final_equity:,.2f}</li>")
    html_parts.append(f"<li><strong>Sharpe Ratio:</strong> {result.sharpe():.2f}</li>")
    html_parts.append(f"<li><strong>Max Drawdown:</strong> {result.max_drawdown:.2%}</li>")
    html_parts.append("</ul>")

    # Equity Curve Plot
    if include_plots:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            result.equity_series.plot(ax=ax)
            ax.set_title("Equity Curve")
            ax.set_xlabel("Date" if isinstance(result.equity_series.index, pd.DatetimeIndex) else "Bar")
            ax.set_ylabel("Equity")
            ax.grid(True)
            img_base64 = _plot_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Equity Curve">')
        except Exception:
            pass

    # Drawdown Plot
    if include_plots:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            drawdown = result.rolling_drawdown("252D" if isinstance(result.equity_series.index, pd.DatetimeIndex) else "252")
            drawdown.plot(ax=ax, color="red")
            ax.set_title("Drawdown")
            ax.set_xlabel("Date" if isinstance(result.equity_series.index, pd.DatetimeIndex) else "Bar")
            ax.set_ylabel("Drawdown")
            ax.grid(True)
            img_base64 = _plot_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Drawdown">')
        except Exception:
            pass

    # Rolling Sharpe
    if include_plots:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            rolling_sharpe = result.rolling_sharpe("63D" if isinstance(result.equity_series.index, pd.DatetimeIndex) else "63")
            rolling_sharpe.plot(ax=ax, color="green")
            ax.set_title("Rolling Sharpe Ratio (63-day window)")
            ax.set_xlabel("Date" if isinstance(result.equity_series.index, pd.DatetimeIndex) else "Bar")
            ax.set_ylabel("Sharpe Ratio")
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax.grid(True)
            img_base64 = _plot_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Rolling Sharpe">')
        except Exception:
            pass

    # Trade Summary
    html_parts.append("<h2>Trade Summary</h2>")
    trade_summary = result.trade_summary()
    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Number of Trades:</strong> {trade_summary['n_trades']}</li>")
    html_parts.append(f"<li><strong>Win Rate:</strong> {trade_summary['win_rate']:.2%}</li>")
    html_parts.append(f"<li><strong>Average Win:</strong> ${trade_summary['avg_win']:.2f}</li>")
    html_parts.append(f"<li><strong>Average Loss:</strong> ${trade_summary['avg_loss']:.2f}</li>")
    html_parts.append(f"<li><strong>Expectancy:</strong> ${trade_summary['expectancy']:.2f}</li>")
    html_parts.append("</ul>")

    # Factor Regression (if provided)
    if factor_returns is not None:
        html_parts.append("<h2>Factor Regression</h2>")
        try:
            factor_result = result.factor_regression(factor_returns)
            html_parts.append("<table>")
            html_parts.append("<tr><th>Factor</th><th>Beta</th><th>t-stat</th></tr>")
            html_parts.append(f"<tr><td>Alpha</td><td>{factor_result.alpha:.4f}</td><td>{factor_result.tstats.get('alpha', 0):.2f}</td></tr>")
            for factor, beta in factor_result.betas.items():
                tstat = factor_result.tstats.get(factor, 0)
                html_parts.append(f"<tr><td>{factor}</td><td>{beta:.4f}</td><td>{tstat:.2f}</td></tr>")
            html_parts.append("</table>")
            html_parts.append(f"<p><strong>R²:</strong> {factor_result.r2:.4f}</p>")
            html_parts.append(f"<p><strong>Observations:</strong> {factor_result.n_obs}</p>")
        except Exception as e:
            html_parts.append(f"<p><em>Factor regression failed: {e}</em></p>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)

