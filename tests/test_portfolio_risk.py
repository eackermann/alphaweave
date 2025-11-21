"""Tests for risk estimation helpers."""

import numpy as np
import pandas as pd

from alphaweave.portfolio.risk import estimate_covariance, estimate_volatility


def test_estimate_covariance_sample():
    """Test sample covariance estimation."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.random.randn(100) * 0.01,
            "B": np.random.randn(100) * 0.01,
        },
        index=dates,
    )

    cov = estimate_covariance(returns, method="sample")

    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (2, 2)
    assert list(cov.index) == ["A", "B"]
    assert list(cov.columns) == ["A", "B"]
    # Covariance matrix should be symmetric
    assert abs(cov.loc["A", "B"] - cov.loc["B", "A"]) < 1e-10


def test_estimate_covariance_ewma():
    """Test EWMA covariance estimation."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.random.randn(100) * 0.01,
            "B": np.random.randn(100) * 0.01,
        },
        index=dates,
    )

    cov = estimate_covariance(returns, method="ewma", span=60)

    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (2, 2)
    assert list(cov.index) == ["A", "B"]
    assert list(cov.columns) == ["A", "B"]
    # Covariance matrix should be symmetric
    assert abs(cov.loc["A", "B"] - cov.loc["B", "A"]) < 1e-10


def test_estimate_volatility():
    """Test volatility estimation."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.random.randn(100) * 0.01,
            "B": np.random.randn(100) * 0.01,
        },
        index=dates,
    )

    vol = estimate_volatility(returns, method="sample", trading_days=252)

    assert isinstance(vol, pd.Series)
    assert len(vol) == 2
    assert "A" in vol.index
    assert "B" in vol.index
    assert all(v >= 0 for v in vol.values)  # Volatility should be non-negative

