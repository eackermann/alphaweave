"""
Example: Cross-sectional momentum strategy using pipeline API.

This strategy demonstrates the pipeline API:
1. Computes momentum factor
2. Screens to top-N assets
3. Normalizes scores to weights
4. Optionally refines with risk parity
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.universe import normalize_scores_to_weights
from alphaweave.pipeline import Pipeline, MomentumFactor, TopN
from alphaweave.strategy.base import Strategy


class CrossSectionalMomentumPipelineStrategy(Strategy):
    """Cross-sectional momentum strategy using pipeline API."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.top_n = 5  # Select top 5 assets
        self.rebalance_freq = "1M"
        self.use_risk_parity = True  # Refine with risk parity

        # Create pipeline
        self.pipeline = Pipeline(
            factors={
                "mom": MomentumFactor(window=63),
            },
            screen=TopN("mom", self.top_n),
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

        # Get momentum scores
        mom_df = result["factors"]["mom"]
        if mom_df.empty:
            return

        # Get last row (current scores)
        scores = mom_df.iloc[-1].dropna()

        if len(scores) == 0:
            return

        # Get screen mask to see which assets are selected
        screen = result["screen"]
        if screen is not None and not screen.empty:
            last_row = screen.iloc[-1]
            selected_assets = [symbol for symbol in last_row.index if last_row[symbol]]
        else:
            # Fallback: select top N by score
            selected_assets = scores.nlargest(self.top_n).index.tolist()

        if len(selected_assets) == 0:
            return

        if self.use_risk_parity:
            # Refine with risk parity on selected assets
            returns_df = self._get_recent_returns(selected_assets, 60)
            if len(returns_df) >= 30:
                cov = estimate_covariance(returns_df, method="ewma", span=60)
                result_opt = risk_parity(cov)
                weights = result_opt.weights
            else:
                # Fallback to normalized scores
                selected_scores = scores[selected_assets]
                weights = normalize_scores_to_weights(selected_scores, long_only=True)
        else:
            # Use normalized momentum scores
            selected_scores = scores[selected_assets]
            weights = normalize_scores_to_weights(selected_scores, long_only=True)

        # Apply weights
        for symbol in self.assets:
            if symbol in weights:
                self.order_target_percent(symbol, weights[symbol])
            else:
                # Exit positions not in top-N
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
        CrossSectionalMomentumPipelineStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

