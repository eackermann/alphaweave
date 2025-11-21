"""
Example: Risk parity portfolio strategy.

This strategy uses risk parity optimization to allocate across
a diversified basket of assets (SPY, TLT, GLD, QQQ).
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.strategy.base import Strategy


class RiskParityPortfolioStrategy(Strategy):
    """Risk parity portfolio allocation strategy."""

    def init(self):
        self.assets = ["SPY", "TLT", "GLD", "QQQ"]
        self.lookback = 60  # days of returns for covariance estimation
        self.rebalance_freq = "1M"  # rebalance monthly

    def _get_recent_returns(self, assets: list[str], lookback: int) -> pd.DataFrame:
        """Get recent returns DataFrame for the specified assets."""
        returns_dict = {}
        current_idx = self._current_index

        if current_idx is None or current_idx < lookback:
            return pd.DataFrame()

        for asset in assets:
            try:
                series = self.series(asset, "close")
                # Get recent prices
                recent_prices = series.iloc[current_idx - lookback : current_idx + 1]
                # Compute returns
                asset_returns = recent_prices.pct_change().dropna()
                if len(asset_returns) > 0:
                    returns_dict[asset] = asset_returns
            except (ValueError, KeyError):
                # Asset not found, skip
                continue

        if not returns_dict:
            return pd.DataFrame()

        # Align returns on common dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        return returns_df

    def next(self, i):
        # Only rebalance on monthly close
        if not (self.schedule.every(self.rebalance_freq) and self.schedule.at_close()):
            return

        # Build returns matrix for lookback window
        returns_df = self._get_recent_returns(self.assets, self.lookback)

        if len(returns_df) < self.lookback // 2:
            return  # Need enough data

        # Estimate covariance matrix
        cov = estimate_covariance(returns_df, method="ewma", span=self.lookback)

        # Run risk parity optimization
        result = risk_parity(cov)
        weights = result.weights  # Series indexed by symbol

        # Apply weights
        for symbol, w in weights.items():
            self.order_target_percent(symbol, w)


def main():
    # Create synthetic data for demonstration
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    import numpy as np

    np.random.seed(42)

    data = {}
    for symbol in ["SPY", "TLT", "GLD", "QQQ"]:
        # Generate correlated price series
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5) + np.linspace(0, 10, 252)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": [1000000] * 252,
            }
        )
        data[symbol] = Frame.from_pandas(df)

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        RiskParityPortfolioStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")


if __name__ == "__main__":
    main()

