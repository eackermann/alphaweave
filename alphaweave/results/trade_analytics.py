"""Trade analytics helper for detailed trade analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeAnalytics:
    """Helper class for trade-level analytics."""

    trades: pd.DataFrame

    def by_symbol(self) -> pd.DataFrame:
        """
        Group stats by symbol: n_trades, win_rate, avg_pnl, etc.

        Returns:
            DataFrame with one row per symbol
        """
        if self.trades.empty:
            return pd.DataFrame()

        grouped = self.trades.groupby("symbol").agg({
            "pnl": ["count", "sum", "mean", lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0],
            "pnl_pct": "mean",
            "duration": "mean",
        })

        grouped.columns = ["n_trades", "total_pnl", "avg_pnl", "win_rate", "avg_pnl_pct", "avg_duration"]
        return grouped

    def by_month(self) -> pd.DataFrame:
        """
        Group trade-level stats by calendar month.

        Returns:
            DataFrame with one row per month
        """
        if self.trades.empty or "entry_time" not in self.trades.columns:
            return pd.DataFrame()

        trades_with_month = self.trades.copy()
        trades_with_month["month"] = pd.to_datetime(trades_with_month["entry_time"]).dt.to_period("M")

        grouped = trades_with_month.groupby("month").agg({
            "pnl": ["count", "sum", "mean", lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0],
            "pnl_pct": "mean",
        })

        grouped.columns = ["n_trades", "total_pnl", "avg_pnl", "win_rate", "avg_pnl_pct"]
        return grouped

    def pnl_curve(self) -> pd.Series:
        """
        Cumulative PnL from trades.

        Returns:
            Series of cumulative PnL
        """
        if self.trades.empty or "pnl" not in self.trades.columns:
            return pd.Series(dtype=float)

        if "exit_time" in self.trades.columns:
            pnl_series = pd.Series(
                self.trades["pnl"].values,
                index=pd.to_datetime(self.trades["exit_time"]),
            )
        else:
            pnl_series = pd.Series(self.trades["pnl"].values)

        return pnl_series.cumsum()

