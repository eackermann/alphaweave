"""Multi-symbol 3-day breakout example with synthetic data."""

import alphaweave as aw
from alphaweave.strategy.base import Strategy
import pandas as pd


# -------------------------------------------------------
# Create synthetic data for 3 symbols
# -------------------------------------------------------

def make_dummy_frame():
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=50, freq="D"),
        "open":   [100 + i for i in range(50)],
        "high":   [101 + i for i in range(50)],
        "low":    [ 99 + i for i in range(50)],
        "close":  [100 + i for i in range(50)],
        "volume": [1000] * 50,
    })
    return aw.Frame.from_pandas(df)


data = {
    "NDX":  make_dummy_frame(),
    "AAPL": make_dummy_frame(),
    "GOOG": make_dummy_frame(),
}


# -------------------------------------------------------
# Strategy: equal-weight breakout across all symbols
# -------------------------------------------------------

class ThreeDayBreakoutEqualWeight(Strategy):

    def __init__(self, data, period=5, streak_len=3):
        super().__init__(data)
        self.period = period
        self.streak_len = streak_len

    def init(self):
        self.symbols = list(self.data.keys())
        self.streak = {sym: 0 for sym in self.symbols}
        self.active = {sym: False for sym in self.symbols}

    def next(self, i):
        # Need enough bars for SMA
        if i < self.period:
            return

        active = []

        # 1) Update streaks + active flags
        for sym in self.symbols:
            price = self.close(sym)
            ma = self.sma(sym, period=self.period)

            if price > ma:
                self.streak[sym] += 1
            else:
                self.streak[sym] = 0

            is_active = self.streak[sym] >= self.streak_len
            self.active[sym] = is_active

            if is_active:
                active.append(sym)

        # 2) Set equal-weight targets
        if len(active) == 0:
            for sym in self.symbols:
                self.order_target_percent(sym, 0.0)
        else:
            w = 1.0 / len(active)
            for sym in self.symbols:
                target = w if self.active[sym] else 0.0
                self.order_target_percent(sym, target)


# -------------------------------------------------------
# Run the backtest
# -------------------------------------------------------

if __name__ == "__main__":
    engine = aw.engine.vector.VectorBacktester()
    result = engine.run(
        ThreeDayBreakoutEqualWeight,
        data=data,
        capital=100_000,     # Same as 100000 — works perfectly
    )

    print("Equity series tail:")
    print(result.equity_series.tail())
    print("Trades:", len(result.trades))

    result.plot_equity()
