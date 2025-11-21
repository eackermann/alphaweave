"""
Example: Generate HTML/Markdown report for a backtest.

This demonstrates how to use the report generator to create
comprehensive backtest reports with metrics, plots, and factor analysis.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.results.report import generate_html_report, generate_markdown_report
from alphaweave.strategy.base import Strategy


class SimpleReportStrategy(Strategy):
    """Simple strategy for report generation example."""

    def init(self):
        pass

    def next(self, i):
        if i == 0:
            self.order_target_percent("_default", 1.0)


def main():
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")  # 1 year of daily data
    import numpy as np
    np.random.seed(42)
    
    # Generate price series with trend
    prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5) + np.linspace(0, 20, 252)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": [1000000] * 252,
    })
    frame = Frame.from_pandas(df)

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        SimpleReportStrategy,
        data=frame,
        capital=100000.0,
    )

    print("Backtest completed!")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe():.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")

    # Generate markdown report
    print("\nGenerating Markdown report...")
    md_report = generate_markdown_report(
        result,
        title="Simple Strategy Backtest Report",
    )
    
    # Save markdown report
    with open("backtest_report.md", "w") as f:
        f.write(md_report)
    print("Markdown report saved to backtest_report.md")

    # Generate HTML report with plots
    print("\nGenerating HTML report...")
    html_report = generate_html_report(
        result,
        title="Simple Strategy Backtest Report",
        include_plots=True,
    )
    
    # Save HTML report
    with open("backtest_report.html", "w") as f:
        f.write(html_report)
    print("HTML report saved to backtest_report.html")

    # Example with factor regression
    print("\nGenerating report with factor regression...")
    
    # Create synthetic benchmark returns
    benchmark_returns = pd.DataFrame({
        "SPY": np.random.randn(252) * 0.01 + 0.0005,  # Slight positive drift
    }, index=dates)

    html_report_with_factors = generate_html_report(
        result,
        title="Simple Strategy Backtest Report (with Factor Analysis)",
        factor_returns=benchmark_returns,
        include_plots=True,
    )
    
    with open("backtest_report_with_factors.html", "w") as f:
        f.write(html_report_with_factors)
    print("HTML report with factors saved to backtest_report_with_factors.html")

    # Show factor regression results
    factor_result = result.factor_regression(benchmark_returns)
    print(f"\nFactor Regression Results:")
    print(f"Alpha: {factor_result.alpha:.4f}")
    print(f"Beta (SPY): {factor_result.betas['SPY']:.4f}")
    print(f"R²: {factor_result.r2:.4f}")


if __name__ == "__main__":
    main()

