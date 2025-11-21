"""
Example: Pipeline with portfolio optimizer integration.

This strategy:
1. Uses pipeline to compute factors and screen universe
2. Applies risk parity optimization on selected assets
3. Demonstrates integration between pipeline and optimizers
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity, min_variance
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.pipeline import (
    Pipeline,
    MomentumFactor,
    LiquidityFilter,
    VolatilityFilter,
    And,
    TopN,
)
from alphaweave.strategy.base import Strategy


class PipelineOptimizerStrategy(Strategy):
    """Strategy combining pipeline screening with portfolio optimization."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.top_n = 5
        self.optimizer = "risk_parity"  # or "min_variance"

        # Create pipeline with filters
        self.pipeline = Pipeline(
            factors={
                "mom": MomentumFactor(window=63),
            },
            filters={
                "liquid": LiquidityFilter(top=500),  # Top 500 by dollar volume
                "low_vol": VolatilityFilter(percentile=50, ascending=True),  # Bottom 50% by volatility
            },
            screen=And(
                "liquid",
                "low_vol",
                TopN("mom", self.top_n),  # Top N by momentum
            ),
        )

    def _get_recent_returns(self, assets: list[str], lookback: int) -> pd.DataFrame:
        """Get recent returns DataFrame."""
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
        if not (self.schedule.every(self.rebalance_freq) and self.schedule.at_close()):
            return

        # Run pipeline
        result = self.run_pipeline(self.pipeline, window="126D")

        # Get screen mask
        screen = result["screen"]
        if screen is None or screen.empty:
            return

        # Get selected assets (last row of screen)
        last_row = screen.iloc[-1]
        selected_assets = [symbol for symbol in last_row.index if last_row[symbol]]

        if len(selected_assets) == 0:
            # Exit all positions
            for symbol in self.assets:
                self.order_target_percent(symbol, 0.0)
            return

        # Get returns for covariance estimation
        returns_df = self._get_recent_returns(selected_assets, 60)
        if len(returns_df) < 30:
            # Fallback to equal weight
            weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}
        else:
            # Estimate covariance
            cov = estimate_covariance(returns_df, method="ewma", span=60)

            # Apply optimizer
            if self.optimizer == "risk_parity":
                opt_result = risk_parity(cov)
            elif self.optimizer == "min_variance":
                opt_result = min_variance(cov)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            weights = opt_result.weights

        # Apply weights
        for symbol in self.assets:
            if symbol in weights:
                self.order_target_percent(symbol, weights[symbol])
            else:
                # Exit positions not selected
                self.order_target_percent(symbol, 0.0)


def main():
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)

    data = {}
    for symbol in ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        # Generate price series with some trend
        trend = np.random.randn() * 0.1
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5 + trend)
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
        PipelineOptimizerStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

