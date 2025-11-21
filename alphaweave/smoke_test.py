"""
Quick smoke test for alphaweave:

- Loads OHLCV data from ./data (CSV/Parquet)
- Runs a simple buy-and-hold strategy on a benchmark symbol
- Runs a 3-day breakout equal-weight strategy on all symbols
- Prints metrics and plots equity curves

Usage:
    python smoke_test.py
"""

import sys
import os

import alphaweave as aw
from alphaweave.data.loaders import load_directory
from alphaweave.strategy.base import Strategy
from alphaweave.engine.vector import VectorBacktester


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------


class BuyAndHoldBenchmark(Strategy):
    """
    Buy-and-hold strategy for a single benchmark symbol.

    On the first bar, go 100% into the benchmark and hold.
    """

    def __init__(self, data, benchmark_symbol: str):
        super().__init__(data)
        self.benchmark_symbol = benchmark_symbol

    def init(self):
        pass

    def next(self, i):
        if i == 0:
            self.order_target_percent(self.benchmark_symbol, 1.0)


class ThreeDayBreakoutEqualWeight(Strategy):
    """
    Multi-symbol equal-weight breakout strategy:

    For each symbol:
      - If close > SMA(period) for `streak_len` consecutive bars, symbol is "active".
      - Allocate 1/K of portfolio to each active symbol (K = number of active symbols).
      - Inactive symbols get 0%.

    Defaults:
      period = 20
      streak_len = 3
    """

    def __init__(self, data, period: int = 20, streak_len: int = 3):
        super().__init__(data)
        self.period = period
        self.streak_len = streak_len

    def init(self):
        self.symbols = list(self.data.keys())
        self.streak = {sym: 0 for sym in self.symbols}
        self.active = {sym: False for sym in self.symbols}

    def next(self, i):
        # Need enough history for SMA
        if i < self.period:
            return

        active_symbols = []

        # 1) Update streaks and active flags
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
                active_symbols.append(sym)

        # 2) Set equal-weight allocations
        k = len(active_symbols)
        if k == 0:
            # Flat everywhere
            for sym in self.symbols:
                self.order_target_percent(sym, 0.0)
        else:
            w = 1.0 / k
            for sym in self.symbols:
                target = w if self.active[sym] else 0.0
                self.order_target_percent(sym, target)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def pick_benchmark_symbol(symbols):
    """Prefer SPY if available, else first symbol alphabetically."""
    symbols = sorted(symbols)
    if "SPY" in symbols:
        return "SPY"
    return symbols[0] if symbols else None


def print_metrics(name: str, result):
    print(f"\n=== {name} ===")
    print(f"Final equity:   {result.final_equity:,.2f}")
    print(f"Total return:   {result.total_return * 100:.2f}%")
    print(f"Max drawdown:   {result.max_drawdown * 100:.2f}%")
    try:
        sharpe = result.sharpe()
    except TypeError:
        # If sharpe is a property in your current version
        sharpe = result.sharpe
    print(f"Sharpe (rf=0):  {sharpe:.3f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, "data")

    if not os.path.isdir(data_dir):
        print(f"ERROR: data directory not found at {data_dir}")
        print("Create a 'data' folder and put your CSV/Parquet OHLCV files there.")
        sys.exit(1)

    print(f"Loading data from {data_dir} ...")
    data = load_directory(data_dir)

    if not data:
        print("ERROR: No symbols loaded from data directory.")
        sys.exit(1)

    symbols = sorted(data.keys())
    print(f"Loaded symbols: {symbols}")

    benchmark = pick_benchmark_symbol(symbols)
    if benchmark is None:
        print("ERROR: Could not determine a benchmark symbol.")
        sys.exit(1)

    print(f"Using benchmark symbol: {benchmark}")

    # --- Buy & hold benchmark ---
    print("\nRunning BuyAndHoldBenchmark...")
    engine = VectorBacktester()
    bh_result = engine.run(
        BuyAndHoldBenchmark,
        data={benchmark: data[benchmark]},
        capital=100_000,
        strategy_kwargs={"benchmark_symbol": benchmark},
    )
    print_metrics("Buy & Hold Benchmark", bh_result)
    try:
        bh_result.plot_equity()
    except Exception as e:
        print(f"(plot_equity failed: {e})")

    # --- 3-day breakout equal-weight across all symbols ---
    print("\nRunning ThreeDayBreakoutEqualWeight on all symbols...")
    engine2 = VectorBacktester()
    breakout_result = engine2.run(
        ThreeDayBreakoutEqualWeight,
        data=data,
        capital=100_000,
        strategy_kwargs={"period": 20, "streak_len": 3},
    )
    print_metrics("3-Day Breakout Equal-Weight", breakout_result)
    try:
        breakout_result.plot_equity()
    except Exception as e:
        print(f"(plot_equity failed: {e})")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
