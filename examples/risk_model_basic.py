"""
Example: Basic Barra-lite risk model construction and usage.

This example demonstrates:
1. Building a risk model from factor data
2. Computing risk decomposition
3. Using risk model in portfolio optimization
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
    decompose_risk,
)
from alphaweave.portfolio.optimizers import min_variance, risk_parity
from alphaweave.pipeline import Pipeline, MomentumFactor, VolatilityFactor
from alphaweave.strategy.base import Strategy


class RiskModelBasicStrategy(Strategy):
    """Basic strategy using Barra-lite risk model."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.lookback = 252

        # Create pipeline for factors
        self.pipeline = Pipeline(
            factors={
                "mom": MomentumFactor(window=63),
                "vol": VolatilityFactor(window=63),
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

        # Run pipeline to get factor data
        result = self.run_pipeline(self.pipeline, window=f"{self.lookback}D")
        factor_data = result["factors"]

        if not factor_data:
            return

        # Build risk model
        # 1. Estimate factor returns
        factor_returns = estimate_factor_returns(returns_df, factor_data, min_assets=5)

        if factor_returns.empty:
            return

        # 2. Estimate factor covariance
        factor_cov = estimate_factor_covariance(factor_returns, method="shrinkage_lw")

        # 3. Compute exposures (current date)
        exposures = compute_exposures(factor_data, date=self.now())

        if exposures.empty:
            return

        # 4. Estimate specific risk
        specific_var = estimate_specific_risk(returns_df, factor_returns, exposures)

        if specific_var.empty:
            return

        # 5. Create risk model
        risk_model = RiskModel(
            exposures=exposures,
            factor_cov=factor_cov,
            specific_var=specific_var,
        )

        # Validate
        try:
            risk_model.validate()
        except ValueError as e:
            print(f"Risk model validation failed: {e}")
            return

        # Use risk model in optimization
        opt_result = min_variance(risk_model=risk_model)
        weights = opt_result.weights

        # Apply weights
        for symbol in self.assets:
            if symbol in weights:
                self.order_target_percent(symbol, weights[symbol])
            else:
                self.order_target_percent(symbol, 0.0)

        # Optional: Decompose risk for analysis
        if len(weights) > 0:
            risk_decomp = decompose_risk(weights, risk_model)
            # Can log risk decomposition if needed
            # print(f"Total vol: {risk_decomp.total_vol:.4f}")


def main():
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)

    data = {}
    for symbol in ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
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
        RiskModelBasicStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

