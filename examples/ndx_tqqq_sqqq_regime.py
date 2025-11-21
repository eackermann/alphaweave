"""NDX regime strategy rotating between TQQQ and SQQQ based on streaks."""

from __future__ import annotations

from dataclasses import dataclass

from alphaweave.data.loaders import load_csv
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy
from alphaweave.utils import ConditionStreak


class NdxTqqqSqqqRegime(Strategy):
    """Regime strategy: long TQQQ or SQQQ based on NDX trend streaks."""

    def init(self) -> None:
        self.above_250_streak = ConditionStreak()
        self.below_250_streak = ConditionStreak()
        self.current_regime = "neutral"

    def _set_regime(self, regime: str) -> None:
        self.current_regime = regime
        if regime == "tqqq":
            self.order_target_percent("TQQQ", 1.0)
            self.order_target_percent("SQQQ", 0.0)
        elif regime == "sqqq":
            self.order_target_percent("TQQQ", 0.0)
            self.order_target_percent("SQQQ", 1.0)
        else:
            self.order_target_percent("TQQQ", 0.0)
            self.order_target_percent("SQQQ", 0.0)

    def next(self, index: int) -> None:
        if index < 250:
            return

        ndx_close = self.close("NDX")
        ma250 = self.sma("NDX", period=250)
        ma10 = self.sma("NDX", period=10)

        above_streak = self.above_250_streak.update(ndx_close > ma250)
        below_streak = self.below_250_streak.update(ndx_close < ma250)

        if above_streak >= 3 and self.current_regime != "tqqq":
            self._set_regime("tqqq")

        if below_streak >= 3 and self.current_regime != "sqqq":
            self._set_regime("sqqq")

        if ndx_close > ma10 and self.current_regime == "sqqq":
            self._set_regime("neutral")


if __name__ == "__main__":
    ndx = load_csv("NDX_daily.csv", symbol="NDX")
    tqqq = load_csv("TQQQ_daily.csv", symbol="TQQQ")
    sqqq = load_csv("SQQQ_daily.csv", symbol="SQQQ")

    data = {"NDX": ndx, "TQQQ": tqqq, "SQQQ": sqqq}

    engine = VectorBacktester()
    result = engine.run(NdxTqqqSqqqRegime, data=data, capital=100_000.0)
    print(result.equity_series.tail())
