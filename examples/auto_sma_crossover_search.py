"""
Example: Automated SMA crossover strategy search.

This example demonstrates:
1. Defining a search space for SMA parameters
2. Creating a strategy factory
3. Running grid search
4. Analyzing results
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.alpha import (
    SearchSpace,
    Param,
    StrategyCandidateSpec,
    EvaluationConfig,
    grid_search,
)
from alphaweave.strategy.base import Strategy


def sma_crossover_factory(params: dict) -> type[Strategy]:
    """Factory function that creates SMA crossover strategy from parameters."""

    fast = params["fast"]
    slow = params["slow"]

    class SMACrossover(Strategy):
        def init(self):
            self.fast = fast
            self.slow = slow
            self.position = 0.0

        def next(self, i):
            if self._current_index is None or self._current_index < self.slow:
                return

            # Get SMA values
            fast_sma = self.sma(period=self.fast)
            slow_sma = self.sma(period=self.slow)

            # Previous values for crossover detection
            if self._current_index > 0:
                prev_fast = self.sma(period=self.fast)
                # Need to get previous bar's SMA - simplified approach
                # In practice, you'd cache these
                try:
                    series = self.series("close")
                    prev_fast_sma = series.iloc[max(0, self._current_index - self.fast) : self._current_index].mean()
                    prev_slow_sma = series.iloc[max(0, self._current_index - self.slow) : self._current_index].mean()
                except:
                    prev_fast_sma = fast_sma
                    prev_slow_sma = slow_sma

                # Crossover logic
                if prev_fast_sma <= prev_slow_sma and fast_sma > slow_sma:
                    # Golden cross: go long
                    self.order_target_percent("_default", 1.0)
                elif prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma:
                    # Death cross: exit
                    self.order_target_percent("_default", 0.0)

    return SMACrossover


def main():
    # Create synthetic data
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    np.random.seed(42)

    # Generate price series with trend
    trend = np.cumsum(np.random.randn(500) * 0.01)
    prices = 100.0 + trend * 10
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
    data = Frame.from_pandas(df)

    # Define search space
    space = SearchSpace(
        params=[
            Param("fast", values=[5, 10, 20]),
            Param("slow", values=[30, 50, 100]),
        ]
    )

    # Create candidate spec
    spec = StrategyCandidateSpec(
        name="SMA_Crossover",
        factory=sma_crossover_factory,
        search_space=space,
    )

    # Evaluation config
    config = EvaluationConfig(
        metric="sharpe",
        time_splits=3,
        min_trades=10,
        overfit_penalty=True,
        stability_penalty=True,
    )

    # Run grid search
    print("Running grid search...")
    results = grid_search(
        candidate=spec,
        data={"_default": data},
        eval_config=config,
        backtester_kwargs={"capital": 100000.0},
        n_jobs=1,  # Use 1 for this example, increase for parallel
    )

    # Display results
    print(f"\nFound {len(results)} configurations")
    print("\nTop 5 results:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Params: {result.params}")
        print(f"   Sharpe: {result.metrics.get('sharpe', 'N/A'):.4f}")
        print(f"   Total Return: {result.metrics.get('total_return', 'N/A'):.2%}")
        print(f"   Trades: {result.metrics.get('n_trades', 'N/A')}")
        if result.diagnostics:
            print(f"   OOS Sharpe: {result.diagnostics.get('out_of_sample_metric', 'N/A'):.4f}")

    # Best configuration
    if results:
        best = results[0]
        print(f"\n\nBest configuration:")
        print(f"  Fast SMA: {best.params['fast']}")
        print(f"  Slow SMA: {best.params['slow']}")
        print(f"  Score: {best.score:.4f}")


if __name__ == "__main__":
    main()

