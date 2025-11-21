"""
Example: Multi-factor composite strategy using pipeline API.

This strategy:
1. Computes multiple factors (momentum, volatility, RSI)
2. Normalizes each factor with z-score
3. Combines factors into composite score
4. Selects top-N assets by composite score
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.universe import normalize_scores_to_weights, top_n_by_score
from alphaweave.pipeline import (
    Pipeline,
    MomentumFactor,
    VolatilityFactor,
    RSIFactor,
    TopN,
)
from alphaweave.strategy.base import Strategy


class MultiFactorCompositeStrategy(Strategy):
    """Multi-factor composite strategy using pipeline."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.top_n = 5

        # Create pipeline with multiple factors
        mom = MomentumFactor(window=63)
        vol = VolatilityFactor(window=63)
        rsi = RSIFactor(period=14)

        # Normalize factors with z-score
        mom_z = mom.zscore()
        vol_z = vol.zscore()
        rsi_z = rsi.zscore()

        # Composite = momentum - volatility + RSI (inverse volatility, higher RSI better)
        # Note: For volatility, lower is better, so we subtract it
        # For RSI, we normalize to [-1, 1] range
        # Since factor-to-factor operations in expressions are limited,
        # we'll compute composite in the strategy instead
        # composite = mom_z - vol_z + (rsi_z - 50) / 50

        self.pipeline = Pipeline(
            factors={
                "mom": mom_z,
                "vol": vol_z,
                "rsi": rsi_z,
            },
        )

    def next(self, i):
        # Rebalance monthly at close
        if not (self.schedule.every(self.rebalance_freq) and self.schedule.at_close()):
            return

        # Run pipeline
        result = self.run_pipeline(self.pipeline, window="126D")

        # Get factor scores
        mom_df = result["factors"]["mom"]
        vol_df = result["factors"]["vol"]
        rsi_df = result["factors"]["rsi"]

        if mom_df.empty or vol_df.empty or rsi_df.empty:
            return

        # Get last row (current scores)
        mom_scores = mom_df.iloc[-1].dropna()
        vol_scores = vol_df.iloc[-1].dropna()
        rsi_scores = rsi_df.iloc[-1].dropna()

        # Align and compute composite
        aligned = pd.DataFrame({
            "mom": mom_scores,
            "vol": vol_scores,
            "rsi": rsi_scores,
        }).dropna()

        if len(aligned) == 0:
            return

        # Composite = momentum - volatility + normalized RSI
        composite = aligned["mom"] - aligned["vol"] + (aligned["rsi"] - 50) / 50

        # Select top N by composite score
        selected_assets = top_n_by_score(composite, self.top_n, ascending=False)

        if len(selected_assets) == 0:
            return

        # Normalize scores to weights
        selected_scores = scores[selected_assets]
        weights = normalize_scores_to_weights(selected_scores, long_only=True)

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
        MultiFactorCompositeStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

