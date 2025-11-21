"""Tests for bootstrap_equity helper."""

import pandas as pd

from alphaweave.analysis import bootstrap_equity
from alphaweave.results.result import BacktestResult


def test_bootstrap_equity_produces_samples():
    equity = pd.Series([100.0, 105.0, 102.0, 110.0, 108.0])
    base_result = BacktestResult(equity_series=equity, trades=[])

    boot = bootstrap_equity(base_result, samples=20)

    assert len(boot.samples) == 20
    assert all(sample.final_equity > 0 for sample in boot.samples)
    assert boot.mean_final_equity > 0
