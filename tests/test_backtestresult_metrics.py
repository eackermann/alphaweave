"""Tests for BacktestResult metrics."""

import pandas as pd

from alphaweave.results.result import BacktestResult


def test_backtestresult_metrics():
    result = BacktestResult([100, 110, 105, 120], trades=[])

    assert result.final_equity == 120
    assert abs(result.total_return - 0.2) < 1e-9

    expected_drawdown = min((100 / 100 - 1), (110 / 110 - 1), (105 / 110 - 1), (120 / 120 - 1))
    assert abs(result.max_drawdown - expected_drawdown) < 1e-9

    sharpe = result.sharpe()
    assert sharpe != 0.0
    assert sharpe == sharpe  # not NaN


def test_sharpe_with_and_without_rf():
    equity = pd.Series([100.0, 105.0, 110.0, 100.0, 120.0])
    result = BacktestResult(equity_series=equity, trades=[])

    sharpe_rf0 = result.sharpe(rf_annual=0.0)
    sharpe_rf5 = result.sharpe(rf_annual=0.05)

    assert sharpe_rf0 != 0.0
    assert sharpe_rf5 != 0.0
    assert sharpe_rf0 != sharpe_rf5


def test_realized_cost_helpers():
    result = BacktestResult([100, 105, 107], trades=[])
    result._trades_df = pd.DataFrame(
        {
            "exit_time": pd.date_range("2024-01-01", periods=3, freq="D"),
            "fees": [1.0, 2.0, 3.0],
            "slippage": [0.5, 0.5, 0.5],
            "size": [100, 200, 100],
        }
    )

    cost_series = result.realized_cost_series()
    assert cost_series.iloc[-1] == sum(result._trades_df["fees"] + result._trades_df["slippage"])

    cost_per_turnover = result.realized_cost_per_turnover()
    turnover = 0.5 * result._trades_df["size"].abs().sum()
    expected = (result._trades_df["fees"].abs().sum() + result._trades_df["slippage"].abs().sum()) / turnover
    assert abs(cost_per_turnover - expected) < 1e-12
