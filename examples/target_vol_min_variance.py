"""
Example: Target volatility with minimum variance base.

This strategy uses minimum variance optimization to get base weights,
then scales them to target a specific volatility level.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import min_variance, target_volatility
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.strategy.base import Strategy


class TargetVolMinVarianceStrategy(Strategy):
    """Target volatility strategy with minimum variance base."""

    def init(self):
        self.assets = ["SPY", "TLT", "GLD"]
        self.lookback = 126  # days for covariance estimation
        self.target_vol = 0.10  # 10% annualized volatility target

    def _get_recent_returns(self, assets: list[str], lookback: int) -> pd.DataFrame:
        """Get recent returns DataFrame for the specified assets."""
        returns_dict = {}
        current_idx = self._current_index

        if current_idx is None or current_idx < lookback:
            return pd.DataFrame()

        for asset in assets:
            try:
                series = self.series(asset, "close")
                recent_prices = series.iloc[current_idx - lookback : current_idx + 1]
                asset_returns = recent_prices.pct_change().dropna()
                if len(asset_returns) > 0:
                    returns_dict[asset] = asset_returns
            except (ValueError, KeyError):
                continue

        if not returns_dict:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        return returns_df

    def next(self, i):
        # Rebalance monthly at close
        if not (self.schedule.every("1M") and self.schedule.at_close()):
            return

        returns_df = self._get_recent_returns(self.assets, self.lookback)
        if len(returns_df) < self.lookback // 2:
            return

        # Estimate covariance
        cov = estimate_covariance(returns_df)

        # Get minimum variance base weights
        minvar_res = min_variance(cov)
        base_weights = minvar_res.weights

        # Scale to target volatility
        target_res = target_volatility(
            base_weights,
            cov,
            target_vol=self.target_vol,
            max_leverage=2.0,
        )

        # Apply scaled weights
        for symbol, w in target_res.weights.items():
            self.order_target_percent(symbol, w)


def main():
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    import numpy as np

    np.random.seed(42)

    data = {}
    for symbol in ["SPY", "TLT", "GLD"]:
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5)
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

    backtester = VectorBacktester()
    result = backtester.run(
        TargetVolMinVarianceStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")


if __name__ == "__main__":
    main()

