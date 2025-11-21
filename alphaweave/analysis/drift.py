"""Utilities for comparing live vs backtest runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from alphaweave.monitoring.run import RunMonitor


def compute_live_drift_series(
    backtest_equity: pd.Series,
    live_run: "RunMonitor",
) -> pd.Series:
    """
    Align backtest equity with live run equity and compute drift series.

    Drift is defined as (live - backtest) / backtest.
    """
    if backtest_equity is None or backtest_equity.empty:
        return pd.Series(dtype=float)

    live_equity = live_run.equity_curve()
    if live_equity.empty:
        return pd.Series(dtype=float)

    backtest = backtest_equity.sort_index()
    live_equity = live_equity.sort_index()
    aligned_backtest = backtest.reindex(live_equity.index, method="ffill").dropna()
    if aligned_backtest.empty:
        return pd.Series(dtype=float)

    aligned_live = live_equity.loc[aligned_backtest.index]
    base = aligned_backtest.replace(0, pd.NA)
    drift = (aligned_live - aligned_backtest) / base
    drift = drift.fillna(0.0)
    drift.name = "drift"
    return drift


