"""
Example: Multi-factor style tilt strategy using expanded factor library.

This strategy demonstrates:
1. Multiple factor types (momentum, volatility, trend)
2. Statistical transforms (winsorization, normalization)
3. Factor combination
4. Integration with portfolio optimizers
"""

import pandas as pd
import numpy as np

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.portfolio.optimizers import risk_parity
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.universe import normalize_scores_to_weights
from alphaweave.pipeline import (
    Pipeline,
    MomentumFactor,
    VolatilityFactor,
    TrendSlopeFactor,
    LowVolatilityFactor,
    LiquidityFilter,
    TopN,
    And,
)
from alphaweave.strategy.base import Strategy


class MultiFactorStyleTiltStrategy(Strategy):
    """Multi-factor strategy with style tilts using expanded factor library."""

    def init(self):
        self.assets = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
        self.rebalance_freq = "1M"
        self.top_n = 5

        # Create factors with transforms
        mom = MomentumFactor(window=63)
        vol = VolatilityFactor(window=63)
        trend = TrendSlopeFactor(window=63)
        low_vol = LowVolatilityFactor(window=63)

        # Apply transforms
        mom_z = mom.zscore()
        vol_z = vol.zscore()
        trend_z = trend.zscore()
        low_vol_z = low_vol.zscore()

        # Composite: momentum + trend - volatility (prefer low vol)
        # Note: We'll compute composite in next() since factor-to-factor ops are limited
        self.pipeline = Pipeline(
            factors={
                "mom": mom_z,
                "vol": vol_z,
                "trend": trend_z,
                "low_vol": low_vol_z,
            },
            filters={
                "liquid": LiquidityFilter(top=500),
            },
            screen=And(
                "liquid",
                TopN("mom", self.top_n * 2),  # Pre-filter by momentum
            ),
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

        # Get factor scores
        mom_df = result["factors"]["mom"]
        vol_df = result["factors"]["vol"]
        trend_df = result["factors"]["trend"]
        low_vol_df = result["factors"]["low_vol"]

        if any(df.empty for df in [mom_df, vol_df, trend_df, low_vol_df]):
            return

        # Get last row (current scores)
        mom_scores = mom_df.iloc[-1].dropna()
        vol_scores = vol_df.iloc[-1].dropna()
        trend_scores = trend_df.iloc[-1].dropna()
        low_vol_scores = low_vol_df.iloc[-1].dropna()

        # Align and compute composite
        aligned = pd.DataFrame({
            "mom": mom_scores,
            "vol": vol_scores,
            "trend": trend_scores,
            "low_vol": low_vol_scores,
        }).dropna()

        if len(aligned) == 0:
            return

        # Composite: momentum + trend - volatility + low_vol preference
        composite = (
            aligned["mom"] * 0.4 +
            aligned["trend"] * 0.3 -
            aligned["vol"] * 0.2 +
            aligned["low_vol"] * 0.1
        )

        # Get screen mask
        screen = result["screen"]
        if screen is not None and not screen.empty:
            last_row = screen.iloc[-1]
            pre_selected = [symbol for symbol in last_row.index if last_row[symbol]]
            # Further filter by composite score
            if len(pre_selected) > 0:
                composite_filtered = composite[composite.index.isin(pre_selected)]
                selected_assets = composite_filtered.nlargest(self.top_n).index.tolist()
            else:
                selected_assets = composite.nlargest(self.top_n).index.tolist()
        else:
            selected_assets = composite.nlargest(self.top_n).index.tolist()

        if len(selected_assets) == 0:
            return

        # Get returns for risk parity
        returns_df = self._get_recent_returns(selected_assets, 60)
        if len(returns_df) >= 30:
            cov = estimate_covariance(returns_df, method="ewma", span=60)
            result_opt = risk_parity(cov)
            weights = result_opt.weights
        else:
            # Fallback to equal weight
            weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}

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
        # Generate price series with different characteristics
        trend = np.random.randn() * 0.1
        vol_mult = 0.5 + np.random.rand() * 0.5
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5 * vol_mult + trend)
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
        MultiFactorStyleTiltStrategy,
        data=data,
        capital=100000.0,
    )

    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

