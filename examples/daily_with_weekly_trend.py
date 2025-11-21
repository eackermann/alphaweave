"""Example: daily execution with weekly trend filter."""

from alphaweave.data.loaders import load_csv
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class DailyWithWeeklyTrend(Strategy):
    """Goes long when daily close is above weekly close."""

    def init(self) -> None:
        self.register_timeframe("1W", "W")

    def next(self, index):
        daily_close = self.close("SPY")
        try:
            weekly_close = self.close("SPY", timeframe="1W")
        except ValueError:
            weekly_close = None

        if weekly_close is None:
            return

        if daily_close > weekly_close:
            self.order_target_percent("SPY", 1.0)
        else:
            self.order_target_percent("SPY", 0.0)


if __name__ == "__main__":
    data = {"SPY": load_csv("SPY_daily.csv", symbol="SPY")}
    engine = VectorBacktester()
    result = engine.run(DailyWithWeeklyTrend, data=data, capital=100_000.0)
    print(result.equity_series.tail())
