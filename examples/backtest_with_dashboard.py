"""
Example: Run a backtest with monitoring and generate an HTML dashboard.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.monitoring.core import InMemoryMonitor
from alphaweave.monitoring.dashboard import generate_html_dashboard
from alphaweave.monitoring.run import RunMonitor
from alphaweave.strategy.base import Strategy


class BalancedStrategy(Strategy):
    def init(self):
        self.assets = list(self.data.keys())

    def next(self, i):
        weight = 1.0 / len(self.assets)
        for symbol in self.assets:
            self.order_target_percent(symbol, weight)
        self.log_metric("rebalance_weight", weight)


def _make_data(symbols, periods=60):
    frames = {}
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    for idx, symbol in enumerate(symbols):
        prices = 100 + idx * 2 + pd.Series(range(periods)) * 0.5
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": 1000 + idx * 10,
            }
        )
        frames[symbol] = Frame.from_pandas(df)
    return frames


def main():
    data = _make_data(["SPY", "TLT", "GLD"])
    monitor = InMemoryMonitor()
    backtester = VectorBacktester()
    result = backtester.run(
        BalancedStrategy,
        data=data,
        capital=100_000.0,
        monitor=monitor,
    )
    run = RunMonitor(monitor)
    html = generate_html_dashboard(run, title="Balanced Strategy Dashboard")
    with open("dashboard.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Final equity: {result.final_equity:,.2f}")
    print("Dashboard saved to dashboard.html")


if __name__ == "__main__":
    main()


