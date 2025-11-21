"""
Example: Monthly rebalance with turnover cap and cost-aware optimizer.

This script demonstrates how to pass transaction cost models and turnover
constraints into the portfolio optimizers without changing the Strategy API.
"""

import numpy as np
import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.costs import ProportionalCostModel
from alphaweave.portfolio.optimizers import mean_variance
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.turnover import TurnoverConstraint
from alphaweave.strategy.base import Strategy


class CostAwareEqualWeight(Strategy):
    """Simple strategy that keeps weights balanced but cost-aware."""

    def init(self):
        self.assets = ["SPY", "TLT", "GLD", "QQQ"]
        self.lookback = 63
        self.prev_weights = pd.Series(0.0, index=self.assets)
        self.turnover_constraint = TurnoverConstraint(max_turnover=0.2)
        self.cost_model = ProportionalCostModel(cost_per_dollar=0.0005)

    def _window_returns(self) -> pd.DataFrame:
        returns = {}
        if self._current_index is None or self._current_index < self.lookback:
            return pd.DataFrame()
        for asset in self.assets:
            prices = self.series(asset, "close")
            window = prices.iloc[self._current_index - self.lookback : self._current_index + 1]
            asset_returns = window.pct_change().dropna()
            if not asset_returns.empty:
                returns[asset] = asset_returns
        return pd.DataFrame(returns).dropna()

    def next(self, index):
        if not (self.schedule.every("1M") and self.schedule.at_close()):
            return

        returns_df = self._window_returns()
        if returns_df.empty:
            return

        cov = estimate_covariance(returns_df)
        mu = returns_df.mean()

        result = mean_variance(
            expected_returns=mu,
            cov_matrix=cov,
            prev_weights=self.prev_weights,
            transaction_cost_model=self.cost_model,
            turnover_constraint=self.turnover_constraint,
        )

        for symbol, weight in result.weights.items():
            self.order_target_percent(symbol, weight)

        self.prev_weights = result.weights


def _generate_synthetic_data(symbols: list[str], periods: int = 252) -> dict[str, Frame]:
    dates = pd.date_range("2023-01-01", periods=periods, freq="B")
    data: dict[str, Frame] = {}
    rng = np.random.default_rng(42)
    for sym in symbols:
        prices = 100 + np.cumsum(rng.normal(0, 0.5, size=len(dates)))
        pdf = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices + 0.5,
                "low": prices - 0.5,
                "close": prices,
                "volume": rng.uniform(1e6, 2e6, size=len(dates)),
            }
        )
        data[sym] = Frame.from_pandas(pdf)
    return data


def main():
    data = _generate_synthetic_data(["SPY", "TLT", "GLD", "QQQ"])
    backtester = VectorBacktester()
    result = backtester.run(
        CostAwareEqualWeight,
        data=data,
        capital=100000.0,
    )
    print("Final equity:", f"${result.final_equity:,.2f}")
    print("Estimated cost per turnover:", result.realized_cost_per_turnover())


if __name__ == "__main__":
    main()


