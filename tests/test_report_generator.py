"""Tests for report generation."""

import pandas as pd

from alphaweave.results.report import generate_html_report, generate_markdown_report
from alphaweave.results.result import BacktestResult


def test_generate_markdown_report():
    """Test markdown report generation."""
    equity = [100.0, 110.0, 120.0]
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    report = generate_markdown_report(result, title="Test Report")
    assert isinstance(report, str)
    assert "Test Report" in report
    assert "Performance Metrics" in report
    assert "Total Return" in report


def test_generate_html_report():
    """Test HTML report generation."""
    equity = [100.0, 110.0, 120.0]
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    report = generate_html_report(result, title="Test Report", include_plots=False)
    assert isinstance(report, str)
    assert "<html>" in report
    assert "Test Report" in report
    assert "Performance Metrics" in report


def test_report_with_factor_returns():
    """Test report generation with factor returns."""
    equity = [100.0 + i * 0.1 for i in range(50)]
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    result = BacktestResult(equity_series=equity, trades=[], timestamps=dates)

    factor_returns = pd.DataFrame({
        "SPY": [0.01] * 50,
    }, index=dates)

    report = generate_markdown_report(result, factor_returns=factor_returns)
    assert "Factor Regression" in report

