"""
Example: Cross-sectional momentum strategy with portfolio optimization.

This strategy:
1. Computes cross-sectional momentum signals
2. Selects top-N assets by momentum
3. Normalizes scores to weights
4. Optionally refines with risk parity
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.universe import normalize_scores_to_weights, top_n_by_score
from alphaweave.strategy.base import Strategy


class CrossSectionalMomentumStrategy(Strategy):
    """Cross-sectional momentum strategy with portfolio construction."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.momentum_lookback = 20  # days for momentum calculation
        self.top_n = 5  # Select top 5 assets
        self.rebalance_freq = "1M"
        self.use_risk_parity = True  # Refine with risk parity

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

    def _compute_momentum_scores(self, assets: list[str]) -> pd.Series:
        """Compute momentum scores for assets."""
        scores = {}
        current_idx = self._current_index

        if current_idx is None or current_idx < self.momentum_lookback:
            return pd.Series(dtype=float)

        for asset in assets:
            try:
                series = self.series(asset, "close")
                if current_idx >= self.momentum_lookback:
                    # Momentum = (current_price / price_n_days_ago) - 1
                    current_price = series.iloc[current_idx]
                    past_price = series.iloc[current_idx - self.momentum_lookback]
                    momentum = (current_price / past_price) - 1.0
                    scores[asset] = momentum
            except (ValueError, KeyError, IndexError):
                continue

        return pd.Series(scores)

    def next(self, i):
        # Rebalance monthly at close
        if not (self.schedule.every(self.rebalance_freq) and self.schedule.at_close()):
            return

        # Compute momentum scores
        momentum_scores = self._compute_momentum_scores(self.assets)

        if len(momentum_scores) == 0:
            return

        # Select top-N assets
        top_assets = top_n_by_score(momentum_scores, self.top_n, ascending=False)

        if len(top_assets) == 0:
            return

        if self.use_risk_parity:
            # Refine with risk parity on selected assets
            returns_df = self._get_recent_returns(top_assets, 60)
            if len(returns_df) >= 30:
                cov = estimate_covariance(returns_df, method="ewma", span=60)
                result = risk_parity(cov)
                weights = result.weights
            else:
                # Fallback to normalized scores
                top_scores = momentum_scores[top_assets]
                weights = normalize_scores_to_weights(top_scores, long_only=True)
        else:
            # Use normalized momentum scores
            top_scores = momentum_scores[top_assets]
            weights = normalize_scores_to_weights(top_scores, long_only=True)

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
    import numpy as np

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
        CrossSectionalMomentumStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

