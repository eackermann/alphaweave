"""
Example: Factor-neutral multi-factor portfolio strategy.

This strategy:
1. Builds a risk model
2. Creates initial portfolio weights
3. Neutralizes exposure to specific factors (e.g., beta-neutral)
4. Optimizes using risk model
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.risk_model import (
    RiskModel,
    estimate_factor_returns,
    estimate_factor_covariance,
    estimate_specific_risk,
    compute_exposures,
    hedge_exposures,
)
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.pipeline import Pipeline, MomentumFactor, VolatilityFactor, BetaFactor
from alphaweave.strategy.base import Strategy


class FactorNeutralStrategy(Strategy):
    """Factor-neutral strategy using risk model."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.lookback = 252
        self.benchmark = "SPY"  # For beta calculation
        self.neutralize_factors = ["beta"]  # Factors to neutralize

        # Create pipeline
        self.pipeline = Pipeline(
            factors={
                "mom": MomentumFactor(window=63),
                "vol": VolatilityFactor(window=63),
                "beta": BetaFactor(self.benchmark, window=252),
            },
        )

    def _get_returns(self, lookback: int) -> pd.DataFrame:
        """Get returns DataFrame."""
        returns_dict = {}
        current_idx = self._current_index

        if current_idx is None or current_idx < lookback:
            return pd.DataFrame()

        for asset in self.assets:
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

        # Get returns
        returns_df = self._get_returns(self.lookback)
        if returns_df.empty or len(returns_df) < 60:
            return

        # Run pipeline
        result = self.run_pipeline(self.pipeline, window=f"{self.lookback}D")
        factor_data = result["factors"]

        if not factor_data:
            return

        # Build risk model
        factor_returns = estimate_factor_returns(returns_df, factor_data, min_assets=5)
        if factor_returns.empty:
            return

        factor_cov = estimate_factor_covariance(factor_returns, method="shrinkage_lw")
        exposures = compute_exposures(factor_data, date=self.now())
        if exposures.empty:
            return

        specific_var = estimate_specific_risk(returns_df, factor_returns, exposures)
        if specific_var.empty:
            return

        risk_model = RiskModel(
            exposures=exposures,
            factor_cov=factor_cov,
            specific_var=specific_var,
        )

        try:
            risk_model.validate()
        except ValueError:
            return

        # Get initial weights (e.g., from risk parity)
        initial_result = risk_parity(risk_model=risk_model)
        initial_weights = initial_result.weights

        # Neutralize factor exposures
        if self.neutralize_factors:
            # Check which factors are available
            available_factors = [f for f in self.neutralize_factors if f in exposures.columns]
            if available_factors:
                adjusted_weights = hedge_exposures(
                    initial_weights,
                    exposures,
                    neutralize_factors=available_factors,
                )
            else:
                adjusted_weights = initial_weights
        else:
            adjusted_weights = initial_weights

        # Apply weights
        for symbol in self.assets:
            if symbol in adjusted_weights:
                self.order_target_percent(symbol, adjusted_weights[symbol])
            else:
                self.order_target_percent(symbol, 0.0)


def main():
    # Create synthetic data including benchmark
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)

    data = {}
    # Add benchmark
    spy_prices = 400.0 + np.cumsum(np.random.randn(252) * 2.0)
    data["SPY"] = Frame.from_pandas(
        pd.DataFrame(
            {
                "datetime": dates,
                "open": spy_prices,
                "high": spy_prices + 2.0,
                "low": spy_prices - 2.0,
                "close": spy_prices,
                "volume": [10000000] * 252,
            }
        )
    )

    # Add assets
    for symbol in ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        trend = np.random.randn() * 0.1
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5 + trend)
        data[symbol] = Frame.from_pandas(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "open": prices,
                    "high": prices + 1.0,
                    "low": prices - 1.0,
                    "close": prices,
                    "volume": [1000000] * 252,
                }
            )
        )

    backtester = VectorBacktester()
    result = backtester.run(
        FactorNeutralStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

