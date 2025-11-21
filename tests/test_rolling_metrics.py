"""Tests for rolling metrics in BacktestResult."""

import pandas as pd

from alphaweave.results.result import BacktestResult


def test_rolling_return():
    """Test rolling return calculation."""
    # Create simple equity series
    equity = [100.0, 110.0, 105.0, 120.0, 130.0]
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    # Test calendar mode
    rolling = result.rolling_return("2D", freq="calendar")
    assert len(rolling) >= 4  # Rolling window may drop some initial values
    assert not rolling.isna().all()

    # Test bars mode
    rolling_bars = result.rolling_return("2", freq="bars")
    assert len(rolling_bars) >= 4  # Rolling window may drop some initial values


def test_rolling_vol():
    """Test rolling volatility calculation."""
    equity = [100.0, 110.0, 105.0, 120.0, 130.0]
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    rolling_vol = result.rolling_vol("2D", freq="calendar", annualize=False)
    assert len(rolling_vol) >= 4  # Rolling window may drop some initial values
    # Check last non-nan value
    last_valid = rolling_vol.dropna()
    if len(last_valid) > 0:
        assert last_valid.iloc[-1] >= 0  # Volatility should be non-negative


def test_rolling_sharpe():
    """Test rolling Sharpe ratio calculation."""
    equity = [100.0, 110.0, 105.0, 120.0, 130.0]
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    rolling_sharpe = result.rolling_sharpe("2D", freq="calendar")
    assert len(rolling_sharpe) >= 4  # Rolling window may drop some initial values
    # Sharpe can be negative, but should be finite
    assert rolling_sharpe.replace([float('inf'), float('-inf')], pd.NA).notna().any()


def test_rolling_drawdown():
    """Test rolling drawdown calculation."""
    equity = [100.0, 110.0, 105.0, 120.0, 130.0]
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    rolling_dd = result.rolling_drawdown("2D", freq="calendar")
    assert len(rolling_dd) == 5
    # Drawdown should be <= 0
    assert (rolling_dd <= 0).all() or rolling_dd.isna().all()

