"""
Example: Multi-factor optimizer with cost and turnover awareness.

Highlights:
    - Composite transaction cost model (spread + proportional).
    - Turnover constraint plus soft rebalance penalty.
    - Strategy API remains init/next/order_target_percent.
"""

import numpy as np
import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.costs import CompositeCostModel, ProportionalCostModel, SpreadBasedCostModel
from alphaweave.portfolio.optimizers import mean_variance
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.turnover import RebalancePenalty, TurnoverConstraint
from alphaweave.strategy.base import Strategy


class CostAwareMultiFactor(Strategy):
    """Toy multi-factor strategy using cost-aware optimizers."""

    def init(self):
        self.assets = ["SPY", "QQQ", "IWM", "EFA", "TLT"]
        self.lookback = 126
        self.prev_weights = pd.Series(0.0, index=self.assets)
        self.turnover_constraint = TurnoverConstraint(max_turnover=0.25, max_change_per_asset=0.08)
        self.rebalance_penalty = RebalancePenalty(lambda_rebalance=2.0)
        self.cost_model = CompositeCostModel(
            components=[
                ProportionalCostModel(cost_per_dollar=0.0004),
                SpreadBasedCostModel(spread_bps=pd.Series(8, index=self.assets), participation_rate=0.5),
            ]
        )

    def _window_prices(self, asset: str) -> pd.Series:
        prices = self.series(asset, "close")
        if self._current_index is None or self._current_index < self.lookback:
            return pd.Series(dtype=float)
        return prices.iloc[self._current_index - self.lookback : self._current_index + 1]

    def _estimate_expected_returns(self) -> pd.Series:
        scores = {}
        for asset in self.assets:
            window = self._window_prices(asset)
            if len(window) < self.lookback // 2:
                continue
            momentum = window.pct_change(21).iloc[-1]
            volatility = window.pct_change().std()
            carry = window.pct_change(63).iloc[-1]
            score = 0.5 * momentum - 0.3 * volatility + 0.2 * carry
            scores[asset] = score
        if not scores:
            return pd.Series(0.0, index=self.assets)
        score_series = pd.Series(scores)
        return score_series / score_series.abs().sum()

    def next(self, index):
        if not (self.schedule.every("1M") and self.schedule.at_close()):
            return

        if self._current_index is None or self._current_index < self.lookback:
            return

        returns = {}
        for asset in self.assets:
            window = self._window_prices(asset)
            if window.empty:
                continue
            asset_returns = window.pct_change().dropna()
            if not asset_returns.empty:
                returns[asset] = asset_returns

        if len(returns) < 2:
            return

        returns_df = pd.DataFrame(returns).dropna()
        cov = estimate_covariance(returns_df, method="ewma", span=63)
        mu = self._estimate_expected_returns()

        result = mean_variance(
            expected_returns=mu,
            cov_matrix=cov,
            prev_weights=self.prev_weights,
            transaction_cost_model=self.cost_model,
            turnover_constraint=self.turnover_constraint,
            rebalance_penalty=self.rebalance_penalty,
        )

        for symbol, weight in result.weights.items():
            self.order_target_percent(symbol, weight)

        self.prev_weights = result.weights


def _make_data(symbols: list[str], periods: int = 504) -> dict[str, Frame]:
    dates = pd.date_range("2022-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(7)
    data: dict[str, Frame] = {}
    for idx, sym in enumerate(symbols):
        drift = 0.05 + 0.01 * idx
        prices = 100 + np.cumsum(rng.normal(drift / 252, 0.01, size=len(dates)))
        pdf = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices + 0.3,
                "low": prices - 0.3,
                "close": prices,
                "volume": rng.uniform(5e5, 2e6, size=len(dates)),
            }
        )
        data[sym] = Frame.from_pandas(pdf)
    return data


def main():
    data = _make_data(["SPY", "QQQ", "IWM", "EFA", "TLT"])
    backtester = VectorBacktester()
    result = backtester.run(
        CostAwareMultiFactor,
        data=data,
        capital=250000.0,
    )
    print("Final equity:", f"${result.final_equity:,.2f}")
    print("Realized cost / turnover:", result.realized_cost_per_turnover())


if __name__ == "__main__":
    main()


