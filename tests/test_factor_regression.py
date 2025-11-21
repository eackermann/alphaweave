"""Tests for factor regression."""

import numpy as np
import pandas as pd

from alphaweave.analysis.factors import FactorRegressionResult, factor_regression


def test_factor_regression_simple():
    """Test factor regression with simple linear relationship."""
    # Create synthetic data: strategy = 2 * factor + noise
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    factor_returns = pd.DataFrame({
        "MKT": np.random.randn(100) * 0.01,
    }, index=dates)

    strategy_returns = 2.0 * factor_returns["MKT"] + np.random.randn(100) * 0.001

    result = factor_regression(strategy_returns, factor_returns)

    assert isinstance(result, FactorRegressionResult)
    assert abs(result.betas["MKT"] - 2.0) < 0.5  # Should be close to 2
    assert result.r2 > 0.5  # Should have decent R²
    assert result.n_obs == 100


def test_factor_regression_multiple_factors():
    """Test factor regression with multiple factors."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    factor_returns = pd.DataFrame({
        "MKT": np.random.randn(100) * 0.01,
        "SMB": np.random.randn(100) * 0.005,
    }, index=dates)

    strategy_returns = (
        1.5 * factor_returns["MKT"] +
        0.5 * factor_returns["SMB"] +
        np.random.randn(100) * 0.001
    )

    result = factor_regression(strategy_returns, factor_returns)

    assert isinstance(result, FactorRegressionResult)
    assert len(result.betas) == 2
    assert "MKT" in result.betas.index
    assert "SMB" in result.betas.index
    assert result.n_obs == 100


def test_factor_regression_no_constant():
    """Test factor regression without constant term."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    factor_returns = pd.DataFrame({
        "MKT": np.random.randn(50) * 0.01,
    }, index=dates)

    strategy_returns = 2.0 * factor_returns["MKT"]

    result = factor_regression(strategy_returns, factor_returns, add_constant=False)

    assert isinstance(result, FactorRegressionResult)
    assert result.alpha == 0.0  # Should be zero when no constant


def test_backtest_result_factor_regression():
    """Test factor regression via BacktestResult."""
    from alphaweave.results.result import BacktestResult

    equity = [100.0 + i * 0.1 for i in range(100)]
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    factor_returns = pd.DataFrame({
        "SPY": np.random.randn(100) * 0.01,
    }, index=dates)

    factor_result = result.factor_regression(factor_returns)
    assert isinstance(factor_result, FactorRegressionResult)

