"""Higher-level helpers built around InMemoryMonitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from alphaweave.monitoring.core import InMemoryMonitor


@dataclass
class RunMonitor:
    """Convenience wrapper around an InMemoryMonitor."""

    monitor: InMemoryMonitor

    @property
    def bars(self) -> pd.DataFrame:
        return self.monitor.bars_df()

    @property
    def trades(self) -> pd.DataFrame:
        return self.monitor.trades_df()

    @property
    def metrics(self) -> pd.DataFrame:
        return self.monitor.metrics_df()

    # Derived metrics -----------------------------------------------------
    def equity_curve(self) -> pd.Series:
        bars = self.bars
        if bars.empty:
            return pd.Series(dtype=float)
        eq = bars.set_index("timestamp")["equity"].astype(float)
        return eq.sort_index()

    def drawdown_curve(self) -> pd.Series:
        equity = self.equity_curve()
        if equity.empty:
            return pd.Series(dtype=float)
        running_max = equity.cummax().replace(0, np.nan)
        drawdown = equity / running_max - 1.0
        drawdown = drawdown.fillna(0.0)
        drawdown.name = "drawdown"
        return drawdown

    def exposure_over_time(self) -> pd.DataFrame:
        bars = self.bars
        if bars.empty:
            return pd.DataFrame()

        records: list[Dict[str, float]] = []
        timestamps = []
        for _, row in bars.iterrows():
            positions = row.get("positions") or {}
            prices = row.get("prices") or {}
            equity = float(row.get("equity") or 0.0)
            exposures: Dict[str, float] = {}
            if equity == 0:
                exposures = {symbol: 0.0 for symbol in positions.keys()}
            else:
                for symbol, size in positions.items():
                    price = prices.get(symbol, 0.0)
                    exposures[symbol] = (size * price) / equity if equity else 0.0
            timestamps.append(row["timestamp"])
            records.append(exposures)

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df = df.fillna(0.0)
        df.insert(0, "timestamp", timestamps)
        df = df.set_index("timestamp").sort_index()
        return df

    def turnover_over_time(self) -> pd.Series:
        exposure = self.exposure_over_time()
        if exposure.empty:
            return pd.Series(dtype=float)
        diffs = exposure.diff().abs().sum(axis=1).fillna(0.0)
        turnover = 0.5 * diffs
        turnover.name = "turnover"
        return turnover

    def cost_over_time(self) -> pd.Series:
        trades = self.trades
        if trades.empty:
            return pd.Series(dtype=float)
        total_cost = trades["fees"].fillna(0.0)
        if "slippage" in trades.columns:
            total_cost = total_cost + trades["slippage"].fillna(0.0)
        series = pd.Series(total_cost.values, index=trades["timestamp"])
        series = series.cumsum()
        series.name = "cost"
        return series


