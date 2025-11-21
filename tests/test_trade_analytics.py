"""Tests for trade analytics in BacktestResult."""

from datetime import datetime

import pandas as pd

from alphaweave.core.types import Fill
from alphaweave.results.result import BacktestResult


def test_trades_dataframe():
    """Test that trades DataFrame is built correctly."""
    equity = [100.0, 110.0, 120.0]
    fills = [
        Fill(order_id=1, symbol="AAPL", size=10.0, price=100.0, datetime=datetime(2023, 1, 1)),
        Fill(order_id=2, symbol="AAPL", size=-10.0, price=110.0, datetime=datetime(2023, 1, 2)),
    ]
    result = BacktestResult(equity_series=equity, trades=fills)

    trades_df = result.trades
    assert isinstance(trades_df, pd.DataFrame)
    assert len(trades_df) > 0 or len(fills) == 0


def test_trade_summary():
    """Test trade summary statistics."""
    equity = [100.0, 110.0, 120.0, 130.0]
    fills = [
        Fill(order_id=1, symbol="AAPL", size=10.0, price=100.0, datetime=datetime(2023, 1, 1)),
        Fill(order_id=2, symbol="AAPL", size=-10.0, price=110.0, datetime=datetime(2023, 1, 2)),
        Fill(order_id=3, symbol="MSFT", size=5.0, price=200.0, datetime=datetime(2023, 1, 3)),
        Fill(order_id=4, symbol="MSFT", size=-5.0, price=210.0, datetime=datetime(2023, 1, 4)),
    ]
    result = BacktestResult(equity_series=equity, trades=fills)

    summary = result.trade_summary()
    assert "n_trades" in summary
    assert "win_rate" in summary
    assert "avg_win" in summary
    assert "avg_loss" in summary
    assert summary["n_trades"] >= 0


def test_trade_distribution():
    """Test trade distribution quantiles."""
    equity = [100.0, 110.0, 120.0]
    fills = [
        Fill(order_id=1, symbol="AAPL", size=10.0, price=100.0, datetime=datetime(2023, 1, 1)),
        Fill(order_id=2, symbol="AAPL", size=-10.0, price=110.0, datetime=datetime(2023, 1, 2)),
    ]
    result = BacktestResult(equity_series=equity, trades=fills)

    dist = result.trade_distribution()
    assert "pnl" in dist
    assert "pnl_pct" in dist
    assert "duration" in dist


def test_trade_analytics_helper():
    """Test TradeAnalytics helper class."""
    equity = [100.0, 110.0, 120.0]
    fills = [
        Fill(order_id=1, symbol="AAPL", size=10.0, price=100.0, datetime=datetime(2023, 1, 1)),
        Fill(order_id=2, symbol="AAPL", size=-10.0, price=110.0, datetime=datetime(2023, 1, 2)),
    ]
    result = BacktestResult(equity_series=equity, trades=fills)

    analytics = result.trade_analytics()
    assert analytics is not None

    # Test by_symbol
    by_symbol = analytics.by_symbol()
    assert isinstance(by_symbol, pd.DataFrame)

    # Test pnl_curve
    pnl_curve = analytics.pnl_curve()
    assert isinstance(pnl_curve, pd.Series)

