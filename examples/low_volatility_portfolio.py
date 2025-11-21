"""
Example: Low volatility portfolio using pipeline API.

This strategy:
1. Computes volatility factor
2. Filters to low volatility stocks
3. Constructs equal-weight or risk-parity portfolio
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.universe import normalize_scores_to_weights
from alphaweave.pipeline import Pipeline, VolatilityFactor, PercentileFilter
from alphaweave.strategy.base import Strategy


class LowVolatilityPortfolioStrategy(Strategy):
    """Low volatility portfolio strategy using pipeline."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.use_risk_parity = True

        # Create pipeline
        self.pipeline = Pipeline(
            factors={
                "vol": VolatilityFactor(window=63),
            },
            screen=PercentileFilter("vol", bottom=30, ascending=True),  # Bottom 30% by volatility
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

        if self.use_risk_parity:
            # Refine with risk parity on selected assets
            returns_df = self._get_recent_returns(selected_assets, 60)
            if len(returns_df) >= 30:
                cov = estimate_covariance(returns_df, method="ewma", span=60)
                result_opt = risk_parity(cov)
                weights = result_opt.weights
            else:
                # Fallback to equal weight
                weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}
        else:
            # Equal weight
            weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}

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
        # Generate price series with different volatilities
        # Some stocks have lower volatility
        vol_mult = 0.3 if symbol in ["AAPL", "MSFT", "GOOG"] else 1.0
        trend = np.random.randn() * 0.05
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5 * vol_mult + trend)
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
        LowVolatilityPortfolioStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

