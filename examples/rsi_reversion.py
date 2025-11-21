"""RSI reversion example using Strategy ergonomics."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.indicators.rsi import RSI
from alphaweave.strategy.base import Strategy


class RsiReversionStrategy(Strategy):
    """Go long when RSI < 30, exit when RSI > 70."""

    def init(self) -> None:
        self.position = 0.0
        self.close_series = self.series("SPY", field="close")
        self.rsi_indicator = RSI(self.close_series, period=14)

    def next(self, index: int) -> None:
        price = self.close("SPY")
        rsi_value = self.rsi_indicator[index]

        if rsi_value < 30 and self.position <= 0:
            self.order_target_percent("SPY", 1.0)
            self.position = 1.0
        elif rsi_value > 70 and self.position > 0:
            self.order_target_percent("SPY", 0.0)
            self.position = 0.0


def build_frame() -> Frame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=250, freq="B"),
            "open": range(250),
            "high": [v + 1 for v in range(250)],
            "low": [max(0, v - 1) for v in range(250)],
            "close": [v + (v % 7) for v in range(250)],
            "volume": [500_000] * 250,
        }
    )
    return Frame.from_pandas(df)


if __name__ == "__main__":
    frame = build_frame()
    data = {"SPY": frame}
    engine = VectorBacktester()
    result = engine.run(RsiReversionStrategy, data=data, capital=25_000.0)
    print(result.equity_series.tail())
