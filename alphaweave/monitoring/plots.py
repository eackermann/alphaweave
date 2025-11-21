"""Matplotlib-based plotting helpers for RunMonitor."""

from __future__ import annotations

from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from alphaweave.monitoring.run import RunMonitor


def plot_equity_and_drawdown(run: RunMonitor):
    """Return a figure with equity on primary axis and drawdown on secondary."""
    equity = run.equity_curve()
    drawdown = run.drawdown_curve()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    if not equity.empty:
        ax1.plot(equity.index, equity.values, label="Equity", color="tab:blue")
        ax1.set_ylabel("Equity", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    if not drawdown.empty:
        ax2.plot(drawdown.index, drawdown.values, label="Drawdown", color="tab:red")
        ax2.set_ylabel("Drawdown", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Equity & Drawdown")
    fig.tight_layout()
    return fig


def plot_exposure_heatmap(run: RunMonitor):
    """Return a heatmap of exposures (weights) over time."""
    exposure = run.exposure_over_time()
    fig, ax = plt.subplots(figsize=(10, 4))
    if exposure.empty:
        ax.text(0.5, 0.5, "No exposure data", ha="center", va="center")
        ax.axis("off")
        return fig

    data = exposure.to_numpy().T
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(range(len(exposure.columns)))
    ax.set_yticklabels(exposure.columns)
    ax.set_title("Exposure Heatmap")
    fig.colorbar(im, ax=ax, label="Weight")
    return fig


def plot_turnover(run: RunMonitor):
    """Plot turnover series."""
    turnover = run.turnover_over_time()
    fig, ax = plt.subplots(figsize=(10, 3))
    if turnover.empty:
        ax.text(0.5, 0.5, "No turnover data", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.plot(turnover.index, turnover.values, color="tab:purple")
    ax.set_title("Turnover Over Time")
    ax.set_ylabel("Turnover")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_trade_pnl_histogram(run: RunMonitor):
    """Plot histogram of realized trade PnL inferred from trade records."""
    trades = run.trades
    fig, ax = plt.subplots(figsize=(6, 4))
    if trades.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        ax.axis("off")
        return fig

    pnls = _compute_trade_pnls(trades)
    if not pnls:
        ax.text(0.5, 0.5, "Insufficient data for PnL", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.hist(pnls, bins=20, color="tab:green", alpha=0.7)
    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def _compute_trade_pnls(trades_df) -> List[float]:
    """Infer realized PnL from sequential trade records assuming FIFO."""
    pnls: List[float] = []
    long_inventory: dict[str, deque] = {}
    short_inventory: dict[str, deque] = {}

    for _, trade in trades_df.iterrows():
        symbol = trade["symbol"]
        size = float(trade["size"])
        price = float(trade["price"])
        long_inventory.setdefault(symbol, deque())
        short_inventory.setdefault(symbol, deque())

        if size > 0:  # buy
            remaining = size
            # Cover shorts first
            queue = short_inventory[symbol]
            while remaining > 0 and queue:
                lot_size, lot_price = queue[0]
                matched = min(lot_size, remaining)
                pnls.append((lot_price - price) * matched)
                lot_size -= matched
                remaining -= matched
                if lot_size <= 1e-12:
                    queue.popleft()
                else:
                    queue[0] = (lot_size, lot_price)
            if remaining > 1e-12:
                long_inventory[symbol].append((remaining, price))
        elif size < 0:  # sell
            remaining = -size
            queue = long_inventory[symbol]
            while remaining > 0 and queue:
                lot_size, lot_price = queue[0]
                matched = min(lot_size, remaining)
                pnls.append((price - lot_price) * matched)
                lot_size -= matched
                remaining -= matched
                if lot_size <= 1e-12:
                    queue.popleft()
                else:
                    queue[0] = (lot_size, lot_price)
            if remaining > 1e-12:
                short_inventory[symbol].append((remaining, price))

    return pnls


