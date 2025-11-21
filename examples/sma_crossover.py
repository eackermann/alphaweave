"""Simple SMA crossover example using Strategy helpers."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class SimpleSmaCrossover(Strategy):
    """Go long when fast SMA crosses above slow SMA."""

    def init(self) -> None:
        self.position = 0.0

    def next(self, index: int) -> None:
        fast = self.sma("SPY", period=20)
        slow = self.sma("SPY", period=50)
        price = self.close("SPY")

        if fast > slow and self.position <= 0:
            self.order_target_percent("SPY", 1.0)
            self.position = 1.0
        elif fast < slow and self.position >= 0:
            self.order_target_percent("SPY", 0.0)
            self.position = 0.0


def build_frame() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=200, freq="B"),
            "open": range(200),
            "high": [v + 1 for v in range(200)],
            "low": [max(0, v - 1) for v in range(200)],
            "close": [v + (v % 5) for v in range(200)],
            "volume": [1_000_000] * 200,
        }
    )
    return Frame.from_pandas(df)


if __name__ == "__main__":
    frame = build_frame()
    data = {"SPY": frame}
    engine = VectorBacktester()
    result = engine.run(SimpleSmaCrossover, data=data, capital=10_000.0)
    print(result.equity_series.tail())
