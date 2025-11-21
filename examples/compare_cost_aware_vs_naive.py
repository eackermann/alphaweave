"""
Example: Compare naive vs. cost-aware optimization outputs.
"""

import numpy as np
import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.costs import ProportionalCostModel
from alphaweave.portfolio.optimizers import mean_variance
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.turnover import RebalancePenalty, TurnoverConstraint
from alphaweave.strategy.base import Strategy


class MeanVarianceComparison(Strategy):
    """Strategy that can toggle cost-awareness via strategy kwargs."""

    def __init__(self, data, *, use_cost: bool = False):
        super().__init__(data)
        self.use_cost = use_cost

    def init(self):
        self.assets = ["SPY", "QQQ", "IWM"]
        self.lookback = 90
        self.prev_weights = pd.Series(0.0, index=self.assets)
        self.cost_model = ProportionalCostModel(cost_per_dollar=0.001) if self.use_cost else None
        self.turnover_constraint = TurnoverConstraint(max_turnover=0.25) if self.use_cost else None
        self.rebalance_penalty = RebalancePenalty(lambda_rebalance=1.5) if self.use_cost else None

    def _returns_window(self) -> pd.DataFrame:
        if self._current_index is None or self._current_index < self.lookback:
            return pd.DataFrame()
        returns = {}
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
        returns_df = self._returns_window()
        if returns_df.empty:
            return

        cov = estimate_covariance(returns_df, method="ewma", span=45)
        mu = returns_df.mean()

        kwargs = {
            "prev_weights": self.prev_weights,
        }
        if self.use_cost:
            kwargs.update(
                {
                    "transaction_cost_model": self.cost_model,
                    "turnover_constraint": self.turnover_constraint,
                    "rebalance_penalty": self.rebalance_penalty,
                }
            )

        result = mean_variance(
            expected_returns=mu,
            cov_matrix=cov,
            **kwargs,
        )

        for symbol, weight in result.weights.items():
            self.order_target_percent(symbol, weight)

        self.prev_weights = result.weights


def _generate_data(symbols: list[str], periods: int = 378) -> dict[str, Frame]:
    dates = pd.date_range("2022-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(5)
    data: dict[str, Frame] = {}
    for idx, sym in enumerate(symbols):
        mu = 0.08 + 0.01 * idx
        sigma = 0.12 + 0.02 * idx
        shocks = rng.normal(mu / 252, sigma / np.sqrt(252), size=len(dates))
        prices = 100 * np.exp(np.cumsum(shocks))
        pdf = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices * (1 + 0.001),
                "low": prices * (1 - 0.001),
                "close": prices,
                "volume": rng.uniform(5e5, 1e6, size=len(dates)),
            }
        )
        data[sym] = Frame.from_pandas(pdf)
    return data


def run_backtests():
    data = _generate_data(["SPY", "QQQ", "IWM"])
    backtester = VectorBacktester()
    naive = backtester.run(MeanVarianceComparison, data=data, capital=150000.0, strategy_kwargs={"use_cost": False})
    cost_aware = backtester.run(MeanVarianceComparison, data=data, capital=150000.0, strategy_kwargs={"use_cost": True})
    return naive, cost_aware


def main():
    naive, cost_aware = run_backtests()
    print("Naive final equity:", f"${naive.final_equity:,.2f}")
    print("Cost-aware final equity:", f"${cost_aware.final_equity:,.2f}")
    print("Naive cost/turnover:", naive.realized_cost_per_turnover())
    print("Cost-aware cost/turnover:", cost_aware.realized_cost_per_turnover())


if __name__ == "__main__":
    main()


