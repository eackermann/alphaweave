"""
Example: Strategies logging custom metrics to the monitor.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.monitoring.core import InMemoryMonitor
from alphaweave.monitoring.run import RunMonitor
from alphaweave.strategy.base import Strategy


class MetricLoggingStrategy(Strategy):
    def init(self):
        self.symbol = next(iter(self.data.keys()))

    def next(self, i):
        prices = self.series(self.symbol, "close")
        momentum = prices.pct_change(5).iloc[self._current_index] if self._current_index >= 5 else 0.0
        target = 0.5 + 0.1 * momentum
        target = max(0.0, min(1.0, target))
        self.log_metric("momentum_signal", float(momentum))
        self.log_metric("target_weight", float(target))
        self.order_target_percent(self.symbol, target)


def main():
    dates = pd.date_range("2024-03-01", periods=40, freq="D")
    prices = 80 + pd.Series(range(40)).rolling(5, min_periods=1).mean()
    df = pd.DataFrame(
        {"datetime": dates, "open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 2000}
    )
    frame = Frame.from_pandas(df)

    monitor = InMemoryMonitor()
    backtester = VectorBacktester()
    backtester.run(
        MetricLoggingStrategy,
        data={"MOM": frame},
        capital=25_000,
        monitor=monitor,
    )
    run = RunMonitor(monitor)
    print(run.metrics.head())


if __name__ == "__main__":
    main()


