"""
Run SPY / NDX / TQQQ / SQQQ strategies with real splits & dividends.

Requirements:
    pip install yfinance plotly

This script:
    - Loads price data from ./data via alphaweave.load_directory()
    - Downloads splits/dividends for SPY, NDX, TQQQ, SQQQ via yfinance
    - Builds CorporateActionsStore
    - Runs the 4 strategies:
        Strat 1: TQQQ trend on NDX 50D SMA (long-only)
        Strat 2: TQQQ/SQQQ regime (50D + 10D pierce)
        Strat 3: Buy & hold TQQQ
        Strat 4: Buy & hold SPY
      both WITH and WITHOUT corporate actions
    - Prints a comparison table
"""

import pandas as pd
import yfinance as yf

import alphaweave as aw
from alphaweave.data.loaders import load_directory
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy
from alphaweave.data.corporate_actions import (
    SplitAction,
    DividendAction,
    build_corporate_actions_store,
)


# ---------------------------------------------------------------------
# Strategies (same as before)
# ---------------------------------------------------------------------

class Strat1_TQQQ_Trend(Strategy):
    """Long-only TQQQ: NDX > 50D SMA => 100% TQQQ, else 0%."""

    def init(self):
        self.ndx_symbol = "NDX"
        self.tqqq_symbol = "TQQQ"
        self.lookback = 50

    def next(self, i):
        if i < self.lookback:
            return

        ndx_close = self.close(self.ndx_symbol)
        sma50 = self.sma(self.ndx_symbol, period=self.lookback)

        if ndx_close > sma50:
            self.order_target_percent(self.tqqq_symbol, 1.0)
        else:
            self.order_target_percent(self.tqqq_symbol, 0.0)


class Strat2_TQQQ_SQQQ_Regime(Strategy):
    """
    Long/short TQQQ/SQQQ:

      1. If NDX daily close > 50D SMA(NDX):
            - long TQQQ 100%, flat SQQQ
      2. If NDX daily close < 50D SMA(NDX):
            - long SQQQ 100%, flat TQQQ
      3. If NDX HIGH > 10D SMA(NDX) (pierce to upside):
            - flatten TQQQ and SQQQ and wait for next 50D-based re-entry.
    """

    def init(self):
        self.ndx_symbol = "NDX"
        self.tqqq_symbol = "TQQQ"
        self.sqqq_symbol = "SQQQ"
        self.slow_period = 50
        self.fast_period = 10

        # cache high series
        self._ndx_high_series = self.series(self.ndx_symbol, "high")

    def next(self, i):
        if i < max(self.slow_period, self.fast_period):
            return

        ndx_close = self.close(self.ndx_symbol)
        sma50 = self.sma(self.ndx_symbol, period=self.slow_period)
        sma10 = self.sma(self.ndx_symbol, period=self.fast_period)

        ndx_high_today = self._ndx_high_series.iloc[i]

        # 3) pierce 10D SMA to upside -> flatten and stop
        if ndx_high_today > sma10:
            self.order_target_percent(self.tqqq_symbol, 0.0)
            self.order_target_percent(self.sqqq_symbol, 0.0)
            return

        # 1) above 50D SMA -> long TQQQ
        if ndx_close > sma50:
            self.order_target_percent(self.tqqq_symbol, 1.0)
            self.order_target_percent(self.sqqq_symbol, 0.0)

        # 2) below 50D SMA -> long SQQQ
        elif ndx_close < sma50:
            self.order_target_percent(self.tqqq_symbol, 0.0)
            self.order_target_percent(self.sqqq_symbol, 1.0)
        # equal -> unchanged


class Strat3_TQQQ_BuyHold(Strategy):
    """Buy & hold TQQQ from first bar."""

    def init(self):
        self.symbol = "TQQQ"

    def next(self, i):
        if i == 0:
            self.order_target_percent(self.symbol, 1.0)


class Strat4_SPY_BuyHold(Strategy):
    """Buy & hold SPY from first bar."""

    def init(self):
        self.symbol = "SPY"

    def next(self, i):
        if i == 0:
            self.order_target_percent(self.symbol, 1.0)


# ---------------------------------------------------------------------
# Corporate actions via yfinance
# ---------------------------------------------------------------------

def fetch_corporate_actions_yf(symbol_map: dict[str, str]):
    """
    Fetch real splits & dividends from yfinance and map back to our symbols.

    symbol_map: mapping from our internal symbol -> yfinance ticker
                e.g. {"SPY": "SPY", "NDX": "^NDX", "TQQQ": "TQQQ", "SQQQ": "SQQQ"}

    Returns (splits_list, dividends_list) where each is a list of
    SplitAction / DividendAction.
    """
    splits: list[SplitAction] = []
    dividends: list[DividendAction] = []

    for sym, yf_sym in symbol_map.items():
        ticker = yf.Ticker(yf_sym)

        # splits: Series indexed by Timestamp, values = ratio
        s = ticker.splits
        if s is not None and len(s) > 0:
            for dt, ratio in s.items():
                splits.append(SplitAction(symbol=sym, date=pd.Timestamp(dt), ratio=float(ratio)))

        # dividends: Series indexed by Timestamp, values = cash per share
        d = ticker.dividends
        if d is not None and len(d) > 0:
            for dt, amount in d.items():
                dividends.append(DividendAction(symbol=sym, date=pd.Timestamp(dt), amount=float(amount)))

    return splits, dividends


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def print_metrics(label: str, result):
    print(f"\n=== {label} ===")
    print(f"Final equity:   {result.final_equity:,.2f}")
    print(f"Total return:   {result.total_return * 100:.2f}%")
    print(f"Max drawdown:   {result.max_drawdown * 100:.2f}%")
    try:
        sharpe = result.sharpe()
    except TypeError:
        sharpe = result.sharpe
    print(f"Sharpe (rf=0):  {sharpe:.3f}")


def compare_results_table(results_no_ca, results_with_ca):
    """
    results_no_ca / results_with_ca: dict[name] = BacktestResult
    """
    rows = []
    for name in results_no_ca.keys():
        r0 = results_no_ca[name]
        r1 = results_with_ca[name]

        def get_sharpe(res):
            val = getattr(res, "sharpe", None)
            return val() if callable(val) else val

        rows.append({
            "strategy": name,
            "final_equity_no_CA": r0.final_equity,
            "final_equity_with_CA": r1.final_equity,
            "total_ret_no_CA_%": r0.total_return * 100.0,
            "total_ret_with_CA_%": r1.total_return * 100.0,
            "max_dd_no_CA_%": r0.max_drawdown * 100.0,
            "max_dd_with_CA_%": r1.max_drawdown * 100.0,
            "sharpe_no_CA": get_sharpe(r0),
            "sharpe_with_CA": get_sharpe(r1),
        })

    df = pd.DataFrame(rows).set_index("strategy")
    print("\n=== Comparison: without vs with corporate actions ===")
    print(df)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # 1) Load price data
    data = load_directory("data")

    required = {"SPY", "NDX", "TQQQ", "SQQQ"}
    missing = required - set(data.keys())
    if missing:
        raise RuntimeError(f"Missing required symbols in data: {missing}")

    # 2) Define strategies
    strategies = [
        ("Strat1_TQQQ_Trend", Strat1_TQQQ_Trend),
        ("Strat2_TQQQ_SQQQ_Regime", Strat2_TQQQ_SQQQ_Regime),
        ("Strat3_TQQQ_BuyHold", Strat3_TQQQ_BuyHold),
        ("Strat4_SPY_BuyHold", Strat4_SPY_BuyHold),
    ]

    engine = VectorBacktester()

    # 3) Run WITHOUT corporate actions (baseline)
    results_no_ca: dict[str, any] = {}

    print("Running baseline (no corporate actions)...")
    for name, strat_cls in strategies:
        res = engine.run(
            strat_cls,
            data=data,
            capital=100_000.0,
        )
        print_metrics(f"{name} (no CA)", res)
        results_no_ca[name] = res

    # 4) Fetch real splits/dividends and build CorporateActionsStore
    symbol_map = {
        "SPY": "SPY",
        "NDX": "^NDX",   # adjust if your NDX data uses a different source
        "TQQQ": "TQQQ",
        "SQQQ": "SQQQ",
    }
    print("\nFetching corporate actions from yfinance...")
    splits, dividends = fetch_corporate_actions_yf(symbol_map)
    ca_store = build_corporate_actions_store(splits=splits, dividends=dividends)
    print(f"Loaded {len(splits)} split events and {len(dividends)} dividend events.")

    # 5) Run WITH corporate actions
    results_with_ca: dict[str, any] = {}

    print("\nRunning with corporate actions...")
    for name, strat_cls in strategies:
        res = engine.run(
            strat_cls,
            data=data,
            capital=100_000.0,
            corporate_actions=ca_store,
        )
        print_metrics(f"{name} (with CA)", res)
        results_with_ca[name] = res

    # 6) Compare
    compare_results_table(results_no_ca, results_with_ca)


if __name__ == "__main__":
    main()
