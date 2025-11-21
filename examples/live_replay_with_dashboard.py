"""
Example: Simulate a live run using LiveEngine and visualize via dashboard.
"""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.live.engine import LiveEngine
from alphaweave.monitoring.core import InMemoryMonitor
from alphaweave.monitoring.dashboard import generate_html_dashboard
from alphaweave.monitoring.run import RunMonitor
from alphaweave.strategy.base import Strategy


class SimpleLiveStrategy(Strategy):
    def init(self):
        self.symbol = next(iter(self.data.keys()))

    def next(self, i):
        target = 0.6 if i % 2 == 0 else 0.4
        self.order_target_percent(self.symbol, target)
        self.log_metric("target_weight", target)


def _make_frame(periods=30):
    dates = pd.date_range("2024-02-01", periods=periods, freq="D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": 50 + pd.Series(range(periods)) * 0.3,
            "high": 51 + pd.Series(range(periods)) * 0.3,
            "low": 49 + pd.Series(range(periods)) * 0.3,
            "close": 50 + pd.Series(range(periods)) * 0.3,
            "volume": 5000,
        }
    )
    return Frame.from_pandas(df)


def main():
    frame = _make_frame()
    monitor = InMemoryMonitor()
    engine = LiveEngine(monitor=monitor)
    engine.run(
        SimpleLiveStrategy,
        data={"LIVE": frame},
        capital=50_000.0,
    )
    run = RunMonitor(monitor)
    html = generate_html_dashboard(run, title="Live Replay Dashboard")
    with open("live_dashboard.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Live-like dashboard saved to live_dashboard.html")


if __name__ == "__main__":
    main()


