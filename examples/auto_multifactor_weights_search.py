"""
Example: Automated multi-factor weight search.

This example demonstrates:
1. Defining continuous parameter search space
2. Creating a multi-factor strategy factory
3. Running random search
4. Finding optimal factor weights
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.alpha import (
    SearchSpace,
    ContinuousParam,
    StrategyCandidateSpec,
    EvaluationConfig,
    random_search,
)
from alphaweave.pipeline import Pipeline, MomentumFactor, VolatilityFactor
from alphaweave.strategy.base import Strategy


def multi_factor_factory(params: dict) -> type[Strategy]:
    """Factory function that creates multi-factor strategy from weight parameters."""

    mom_weight = params["mom_weight"]
    vol_weight = params["vol_weight"]

    class MultiFactorStrategy(Strategy):
        def init(self):
            self.mom_weight = mom_weight
            self.vol_weight = vol_weight
            self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]

            # Create pipeline
            self.pipeline = Pipeline(
                factors={
                    "mom": MomentumFactor(window=63),
                    "vol": VolatilityFactor(window=63),
                },
            )

        def next(self, i):
            if not self.schedule.every("1M"):
                return

            # Run pipeline
            result = self.run_pipeline(self.pipeline, window="126D")

            # Get factor scores
            mom_df = result["factors"]["mom"]
            vol_df = result["factors"]["vol"]

            if mom_df.empty or vol_df.empty:
                return

            # Get last row (current scores)
            mom_scores = mom_df.iloc[-1].dropna()
            vol_scores = vol_df.iloc[-1].dropna()

            # Align and compute composite
            aligned = pd.DataFrame({
                "mom": mom_scores,
                "vol": vol_scores,
            }).dropna()

            if len(aligned) == 0:
                return

            # Composite score
            composite = self.mom_weight * aligned["mom"] - self.vol_weight * aligned["vol"]

            # Select top 5
            top_assets = composite.nlargest(5).index.tolist()

            # Equal weight
            weight_per_asset = 1.0 / len(top_assets) if top_assets else 0.0

            # Apply weights
            for symbol in self.assets:
                if symbol in top_assets:
                    self.order_target_percent(symbol, weight_per_asset)
                else:
                    self.order_target_percent(symbol, 0.0)

    return MultiFactorStrategy


def main():
    # Create synthetic data
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    np.random.seed(42)

    data = {}
    for symbol in ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        trend = np.random.randn() * 0.1
        prices = 100.0 + np.cumsum(np.random.randn(500) * 0.5 + trend)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": [1000000] * 500,
            }
        )
        data[symbol] = Frame.from_pandas(df)

    # Define search space (continuous parameters)
    space = SearchSpace(
        params=[
            ContinuousParam("mom_weight", -1.0, 1.0),
            ContinuousParam("vol_weight", -1.0, 1.0),
        ]
    )

    # Create candidate spec
    spec = StrategyCandidateSpec(
        name="MultiFactor",
        factory=multi_factor_factory,
        search_space=space,
    )

    # Evaluation config
    config = EvaluationConfig(
        metric="sharpe",
        time_splits=3,
        min_trades=20,
        overfit_penalty=True,
        stability_penalty=True,
    )

    # Run random search
    print("Running random search...")
    results = random_search(
        candidate=spec,
        data=data,
        eval_config=config,
        n_samples=50,
        backtester_kwargs={"capital": 100000.0},
        n_jobs=1,
        random_state=42,
    )

    # Display results
    print(f"\nFound {len(results)} configurations")
    print("\nTop 5 results:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Mom Weight: {result.params['mom_weight']:.3f}")
        print(f"   Vol Weight: {result.params['vol_weight']:.3f}")
        print(f"   Sharpe: {result.metrics.get('sharpe', 'N/A'):.4f}")
        print(f"   Total Return: {result.metrics.get('total_return', 'N/A'):.2%}")
        print(f"   Trades: {result.metrics.get('n_trades', 'N/A')}")

    # Best configuration
    if results:
        best = results[0]
        print(f"\n\nBest configuration:")
        print(f"  Momentum Weight: {best.params['mom_weight']:.3f}")
        print(f"  Volatility Weight: {best.params['vol_weight']:.3f}")
        print(f"  Score: {best.score:.4f}")


if __name__ == "__main__":
    main()

