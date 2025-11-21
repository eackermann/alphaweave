"""Backtest result container."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphaweave.core.types import Fill

from alphaweave.results.trade_analytics import TradeAnalytics


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

    def __init__(
        self,
        equity_series: List[float],
        trades: List[Any],
        timestamps: Optional[List[Any]] = None,
    ):
        """
        Initialize BacktestResult.

        Args:
            equity_series: List of equity values
            trades: List of Fill objects or trade records
            timestamps: Optional list of timestamps for equity_series
        """
        if timestamps is not None:
            self.equity_series = pd.Series(equity_series, index=timestamps)
        else:
            self.equity_series = pd.Series(equity_series)
        self._trades_raw = trades
        self._trades_df: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.Series] = None

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

    @property
    def returns(self) -> pd.Series:
        """Returns series (pct_change of equity)."""
        if self._returns is None:
            self._returns = self.equity_series.pct_change().dropna()
        return self._returns

    def rolling_return(
        self,
        window: str = "63D",
        freq: Literal["calendar", "bars"] = "calendar",
    ) -> pd.Series:
        """
        Rolling cumulative return over a lookback window.

        Args:
            window: Window size (e.g., "63D" for calendar, "63" for bars)
            freq: "calendar" for time-based or "bars" for bar-count

        Returns:
            Series of rolling returns
        """
        if freq == "calendar":
            if not isinstance(self.equity_series.index, pd.DatetimeIndex):
                raise ValueError("equity_series must have DatetimeIndex for calendar frequency")
            returns = self.returns
            rolling = returns.rolling(window=window)
            return rolling.apply(lambda x: (1 + x).prod() - 1.0)
        else:
            # Bar-count mode
            window_int = int(window) if isinstance(window, str) else window
            returns = self.returns
            rolling = returns.rolling(window=window_int)
            return rolling.apply(lambda x: (1 + x).prod() - 1.0)

    def rolling_vol(
        self,
        window: str = "63D",
        freq: Literal["calendar", "bars"] = "calendar",
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Rolling volatility of returns.

        Args:
            window: Window size
            freq: "calendar" or "bars"
            annualize: If True, annualize the volatility
            trading_days: Trading days per year for annualization

        Returns:
            Series of rolling volatilities
        """
        if freq == "calendar":
            if not isinstance(self.equity_series.index, pd.DatetimeIndex):
                raise ValueError("equity_series must have DatetimeIndex for calendar frequency")
            returns = self.returns
            rolling_std = returns.rolling(window=window).std()
        else:
            window_int = int(window) if isinstance(window, str) else window
            returns = self.returns
            rolling_std = returns.rolling(window=window_int).std()

        if annualize:
            # Approximate periods per year from window
            if freq == "calendar":
                # Parse window to estimate periods per year
                if "D" in window:
                    days = int(window.rstrip("D"))
                    periods_per_year = trading_days / days * trading_days
                elif "W" in window:
                    periods_per_year = 52
                elif "M" in window:
                    periods_per_year = 12
                elif "Y" in window:
                    periods_per_year = 1
                else:
                    periods_per_year = trading_days
            else:
                # For bar-count, assume daily bars
                periods_per_year = trading_days
            return rolling_std * np.sqrt(periods_per_year)
        return rolling_std

    def rolling_sharpe(
        self,
        window: str = "63D",
        freq: Literal["calendar", "bars"] = "calendar",
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Rolling Sharpe ratio over the given window.

        Args:
            window: Window size
            freq: "calendar" or "bars"
            risk_free_rate: Annual risk-free rate
            trading_days: Trading days per year

        Returns:
            Series of rolling Sharpe ratios
        """
        returns = self.returns
        rf_period = risk_free_rate / trading_days if freq == "bars" else risk_free_rate / trading_days

        if freq == "calendar":
            if not isinstance(self.equity_series.index, pd.DatetimeIndex):
                raise ValueError("equity_series must have DatetimeIndex for calendar frequency")
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            periods_per_year = trading_days
        else:
            window_int = int(window) if isinstance(window, str) else window
            rolling_mean = returns.rolling(window=window_int).mean()
            rolling_std = returns.rolling(window=window_int).std()
            periods_per_year = trading_days

        excess_returns = rolling_mean - rf_period
        sharpe = excess_returns / rolling_std * np.sqrt(periods_per_year)
        return sharpe.replace([np.inf, -np.inf], np.nan)

    def rolling_drawdown(
        self,
        window: str = "252D",
        freq: Literal["calendar", "bars"] = "calendar",
    ) -> pd.Series:
        """
        Rolling max drawdown.

        Args:
            window: Window size
            freq: "calendar" or "bars"

        Returns:
            Series of rolling max drawdowns
        """
        equity = self.equity_series

        if freq == "calendar":
            if not isinstance(equity.index, pd.DatetimeIndex):
                raise ValueError("equity_series must have DatetimeIndex for calendar frequency")
            rolling_max = equity.rolling(window=window).max()
        else:
            window_int = int(window) if isinstance(window, str) else window
            rolling_max = equity.rolling(window=window_int).max()

        drawdown = equity / rolling_max - 1.0
        return drawdown

    @property
    def trades(self) -> pd.DataFrame:
        """
        Return a trades DataFrame with standardized columns.

        Builds from fills if not already cached.
        """
        if self._trades_df is not None:
            return self._trades_df

        if not self._trades_raw:
            # Return empty DataFrame with expected columns
            self._trades_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "size",
                    "pnl",
                    "pnl_pct",
                    "duration",
                    "fees",
                    "slippage",
                    "direction",
                ]
            )
            return self._trades_df

        # Build trades from fills
        fills = [f for f in self._trades_raw if isinstance(f, Fill)]
        if not fills:
            # If no Fill objects, return empty DataFrame
            self._trades_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "size",
                    "pnl",
                    "pnl_pct",
                    "duration",
                    "fees",
                    "slippage",
                    "direction",
                ]
            )
            return self._trades_df

        # Group fills by symbol and order_id to reconstruct trades
        # Simple approach: pair buys and sells
        trades_list = []
        positions: Dict[str, List[Fill]] = {}

        for fill in sorted(fills, key=lambda f: f.datetime):
            symbol = fill.symbol
            if symbol not in positions:
                positions[symbol] = []

            # Simple pairing: if size has opposite sign, close position
            if positions[symbol]:
                # Check if this closes a position
                last_fill = positions[symbol][-1]
                if (last_fill.size > 0 and fill.size < 0) or (last_fill.size < 0 and fill.size > 0):
                    # Close position
                    entry_fill = positions[symbol].pop(0) if positions[symbol] else last_fill
                    exit_fill = fill

                    entry_size = abs(entry_fill.size)
                    exit_size = abs(exit_fill.size)
                    trade_size = min(entry_size, exit_size)

                    pnl = (exit_fill.price - entry_fill.price) * trade_size * (1 if entry_fill.size > 0 else -1)
                    pnl_pct = (exit_fill.price / entry_fill.price - 1.0) * (1 if entry_fill.size > 0 else -1)

                    duration = (exit_fill.datetime - entry_fill.datetime).total_seconds() / 86400  # days

                    trades_list.append({
                        "symbol": symbol,
                        "entry_time": entry_fill.datetime,
                        "exit_time": exit_fill.datetime,
                        "entry_price": entry_fill.price,
                        "exit_price": exit_fill.price,
                        "size": trade_size * (1 if entry_fill.size > 0 else -1),
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "duration": duration,
                        "fees": 0.0,  # Would need to track from engine
                        "slippage": 0.0,  # Would need to track from engine
                        "direction": "long" if entry_fill.size > 0 else "short",
                    })
                else:
                    # Add to position
                    positions[symbol].append(fill)
            else:
                # New position
                positions[symbol].append(fill)

        self._trades_df = pd.DataFrame(trades_list)
        return self._trades_df

    def trade_summary(self) -> Dict[str, Any]:
        """
        Basic summary stats for trades.

        Returns:
            Dictionary with n_trades, win_rate, avg_win, avg_loss, etc.
        """
        trades_df = self.trades
        if trades_df.empty:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "median_duration": 0.0,
            }

        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0.0
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0.0
        expectancy = trades_df["pnl"].mean() if n_trades > 0 else 0.0

        # Consecutive wins/losses
        trades_df_sorted = trades_df.sort_values("entry_time")
        wins = (trades_df_sorted["pnl"] > 0).astype(int)
        losses = (trades_df_sorted["pnl"] < 0).astype(int)

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for w, l in zip(wins, losses):
            if w:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif l:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        median_duration = trades_df["duration"].median() if "duration" in trades_df.columns else 0.0

        return {
            "n_trades": n_trades,
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "median_duration": float(median_duration),
        }

    def trade_distribution(self) -> Dict[str, Any]:
        """
        Return distributions or precomputed quantiles for trade metrics.

        Returns:
            Dictionary with quantiles for pnl, pnl_pct, duration
        """
        trades_df = self.trades
        if trades_df.empty:
            return {
                "pnl": {},
                "pnl_pct": {},
                "duration": {},
            }

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        return {
            "pnl": trades_df["pnl"].quantile(quantiles).to_dict() if "pnl" in trades_df.columns else {},
            "pnl_pct": trades_df["pnl_pct"].quantile(quantiles).to_dict() if "pnl_pct" in trades_df.columns else {},
            "duration": trades_df["duration"].quantile(quantiles).to_dict() if "duration" in trades_df.columns else {},
        }

    def turnover(
        self,
        freq: str = "1M",
    ) -> pd.Series:
        """
        Portfolio turnover per period.

        Turnover_t = 0.5 * sum(|Δweight_i,t|) over i.

        Args:
            freq: Resampling frequency (e.g., "1M", "1D")

        Returns:
            Series indexed by period end timestamp
        """
        trades_df = self.trades
        if trades_df.empty or not isinstance(trades_df.index, pd.DatetimeIndex):
            # If no datetime index, create one from entry_time
            if "entry_time" in trades_df.columns:
                trades_df = trades_df.set_index("entry_time")
            else:
                return pd.Series(dtype=float)

        # Simple approximation: sum absolute trade sizes per period
        if "size" in trades_df.columns:
            trades_df["abs_size"] = trades_df["size"].abs()
            # Handle deprecated 'M' frequency
            resample_freq = freq.replace("M", "ME") if freq == "1M" else freq
            turnover = trades_df["abs_size"].resample(resample_freq).sum() * 0.5
        else:
            turnover = pd.Series(dtype=float)

        return turnover

    def average_slippage_per_share(self) -> float:
        """Average slippage per share across all trades, if available."""
        trades_df = self.trades
        if trades_df.empty or "slippage" not in trades_df.columns:
            return 0.0
        slippage = trades_df["slippage"].sum()
        total_shares = trades_df["size"].abs().sum()
        return float(slippage / total_shares) if total_shares > 0 else 0.0

    def slippage_cost_series(self) -> pd.Series:
        """
        Time series of cumulative slippage cost.

        Returns:
            Series of cumulative slippage
        """
        trades_df = self.trades
        if trades_df.empty or "slippage" not in trades_df.columns:
            return pd.Series(dtype=float)

        if "exit_time" in trades_df.columns:
            slippage_series = pd.Series(
                trades_df["slippage"].values,
                index=pd.to_datetime(trades_df["exit_time"]),
            )
        else:
            slippage_series = pd.Series(trades_df["slippage"].values)

        return slippage_series.cumsum()

    def fee_cost_series(self) -> pd.Series:
        """
        Time series of cumulative fees/commissions.

        Returns:
            Series of cumulative fees
        """
        trades_df = self.trades
        if trades_df.empty or "fees" not in trades_df.columns:
            return pd.Series(dtype=float)

        if "exit_time" in trades_df.columns:
            fee_series = pd.Series(
                trades_df["fees"].values,
                index=pd.to_datetime(trades_df["exit_time"]),
            )
        else:
            fee_series = pd.Series(trades_df["fees"].values)

        return fee_series.cumsum()

    def realized_cost_series(self) -> pd.Series:
        """
        Cumulative realized execution costs (fees + slippage).

        Returns:
            Series of cumulative costs indexed by trade exit time when available.
        """
        trades_df = self.trades
        if trades_df.empty:
            return pd.Series(dtype=float)

        cost_components = []
        for column in ("fees", "slippage"):
            if column in trades_df.columns:
                cost_components.append(trades_df[column])

        if not cost_components:
            return pd.Series(dtype=float)

        total_cost = sum(cost_components)
        if "exit_time" in trades_df.columns:
            index = pd.to_datetime(trades_df["exit_time"])
        elif "entry_time" in trades_df.columns:
            index = pd.to_datetime(trades_df["entry_time"])
        else:
            index = pd.RangeIndex(len(total_cost))

        cost_series = pd.Series(total_cost.values, index=index)
        return cost_series.cumsum()

    def realized_cost_per_turnover(self) -> float:
        """
        Estimate realized cost per unit of turnover based on historical trades.
        """
        trades_df = self.trades
        if trades_df.empty:
            return 0.0

        total_cost = 0.0
        if "fees" in trades_df.columns:
            total_cost += trades_df["fees"].abs().sum()
        if "slippage" in trades_df.columns:
            total_cost += trades_df["slippage"].abs().sum()

        if total_cost == 0:
            return 0.0

        if "size" not in trades_df.columns:
            return 0.0

        total_turnover = trades_df["size"].abs().sum()
        if total_turnover == 0:
            return 0.0

        return float(total_cost / (0.5 * total_turnover))

    def factor_regression(
        self,
        factor_returns: pd.DataFrame,
        **kwargs,
    ) -> Any:
        """
        Run factor regression on strategy returns.

        Args:
            factor_returns: DataFrame with factor returns (columns = factors, index = datetime)
            **kwargs: Additional arguments for factor_regression

        Returns:
            FactorRegressionResult
        """
        from alphaweave.analysis.factors import factor_regression

        return factor_regression(self.returns, factor_returns, **kwargs)

    def trade_analytics(self) -> TradeAnalytics:
        """
        Get TradeAnalytics helper for detailed trade analysis.

        Returns:
            TradeAnalytics instance
        """
        return TradeAnalytics(self.trades)

    def plot_equity(self) -> None:
        """Minimal plotting helper using matplotlib."""
        plt.figure(figsize=(10, 6))
        self.equity_series.plot()
        plt.title("Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()
