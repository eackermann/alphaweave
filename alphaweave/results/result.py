"""Backtest result container."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sharpe_ratio(
    equity_series: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio of the equity curve.

    Args:
        equity_series: Equity curve as pandas Series.
        rf_annual: Annual risk-free rate (e.g., 0.05 for 5%).
        periods_per_year: Number of periods per year (default 252).
    """
    returns = equity_series.pct_change().dropna()
    if returns.empty:
        return 0.0

    std = returns.std()
    if std == 0:
        return 0.0

    rf_period = rf_annual / periods_per_year
    excess_returns = returns - rf_period

    return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, equity_series: List[float], trades: List[Any]):
        self.equity_series = pd.Series(equity_series)
        self.trades = trades

    @property
    def equity_curve(self) -> pd.Series:
        """Alias for backward compatibility and intuitive naming."""
        return self.equity_series

    @property
    def final_equity(self) -> float:
        return float(self.equity_series.iloc[-1])

    @property
    def total_return(self) -> float:
        start = float(self.equity_series.iloc[0])
        end = float(self.equity_series.iloc[-1])
        if start == 0:
            return 0.0
        return (end / start) - 1.0

    @property
    def max_drawdown(self) -> float:
        eq = self.equity_series
        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        return float(drawdown.min())

    def sharpe(self, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Annualized Sharpe ratio using an optional risk-free rate.

        Args:
            rf_annual: Annualized risk-free rate as a decimal.
            periods_per_year: Number of periods per year (default 252).
        """
        return sharpe_ratio(
            self.equity_series,
            rf_annual=rf_annual,
            periods_per_year=periods_per_year,
        )

    @property
    def sharpe_value(self) -> float:
        """Backward-compatible property style Sharpe ratio (rf=0)."""
        return self.sharpe()

    def plot_equity(self) -> None:
        """Minimal plotting helper using matplotlib."""
        plt.figure(figsize=(10, 6))
        self.equity_series.plot()
        plt.title("Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()
