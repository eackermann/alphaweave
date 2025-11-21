"""Tests for corporate actions (splits and dividends)."""

import pandas as pd
from datetime import datetime

from alphaweave.core.frame import Frame
from alphaweave.data.corporate_actions import (
    SplitAction,
    DividendAction,
    CorporateActionsStore,
    load_splits_csv,
    load_dividends_csv,
    build_corporate_actions_store,
)
from alphaweave.engine.portfolio import Portfolio
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class BuyAndHold(Strategy):
    """Simple buy-and-hold strategy for testing."""

    def init(self):
        """Initialize strategy."""
        pass

    def next(self, i):
        """Buy and hold on first bar."""
        if i == 0:
            # Buy 100% on first bar
            if isinstance(self.data, Frame):
                self.order_target_percent("_default", 1.0)
            else:
                # Use first symbol in dict
                symbol = next(iter(self.data.keys()))
                self.order_target_percent(symbol, 1.0)


def test_split_preserves_equity():
    """
    Test that stock splits preserve portfolio equity.

    Create synthetic data with a 2-for-1 split in the middle.
    Strategy buys and holds.
    Equity should not show an artificial drop on the split date.
    """
    # Create synthetic data: 20 days, price goes from 100 to 120
    # Split occurs on day 10 (2-for-1), so price should drop from ~110 to ~55
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    
    # Pre-split: days 0-9, price 100 to 109
    # Post-split: days 10-19, price 55 to 64 (half of 110-128)
    prices = []
    for i in range(20):
        if i < 10:
            price = 100.0 + i
        else:
            # After split, price is roughly half
            price = (100.0 + i) / 2.0
        prices.append(price)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    
    # Create a 2-for-1 split on day 10 (2021-01-11)
    split_date = datetime(2021, 1, 11)
    split = SplitAction(symbol="ASSET", date=split_date, ratio=2.0)
    store = build_corporate_actions_store(splits=[split])
    
    # Run backtest with corporate actions
    backtester = VectorBacktester()
    result_with_split = backtester.run(
        BuyAndHold,
        data={"ASSET": frame},
        capital=10000.0,
        corporate_actions=store,
    )
    
    # Run backtest without corporate actions (for comparison)
    result_without_split = backtester.run(
        BuyAndHold,
        data={"ASSET": frame},
        capital=10000.0,
    )
    
    # Equity should be smooth - no artificial drop on split date
    # With proper split handling, equity should continue growing
    # Check that equity on day 9 (before split) and day 10 (after split) are similar
    equity_before_split = result_with_split.equity_series[9]
    equity_after_split = result_with_split.equity_series[10]
    
    # Equity should be approximately the same (within small tolerance for price movement)
    # The split itself shouldn't cause a drop - only price movement should affect equity
    # Since price goes from 109 to 55 (but we have 2x shares), equity should be similar
    # Actually, if we bought at day 0 at price 100, and on day 9 price is 109:
    #   - Shares: 10000 / 100 = 100 shares
    #   - Equity: 100 * 109 = 10900
    # On day 10, after split:
    #   - Shares: 100 * 2 = 200 shares
    #   - Price: 55 (half of 110)
    #   - Equity: 200 * 55 = 11000
    # So equity should actually increase slightly due to price movement
    
    # The key test: equity should not drop dramatically on split date
    # It should be smooth, only affected by price movement
    assert equity_after_split > 0, "Equity should be positive after split"
    
    # Equity should be approximately maintained (within 5% tolerance for price movement)
    # Since price moves from 109 to 55 but we have 2x shares, equity should be similar
    ratio = equity_after_split / equity_before_split if equity_before_split > 0 else 1.0
    assert 0.95 <= ratio <= 1.05, f"Equity should be preserved across split, got ratio {ratio}"
    
    # Final equity should be positive and reasonable
    assert result_with_split.equity_series.iloc[-1] > 0
    assert len(result_with_split.equity_series) == 20


def test_dividend_increases_cash_equity():
    """
    Test that dividends increase cash and equity.

    Create synthetic data with flat price.
    Strategy buys and holds 100 shares.
    Add a dividend of 1.0 per share on a given date.
    Equity with dividends should be higher than equity without dividends
    by approximately 100 (ignoring fees/slippage).
    """
    # Create synthetic data: 20 days, flat price at 100
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [10000] * 20,
    })
    
    frame = Frame.from_pandas(df)
    
    # Create a dividend of 1.0 per share on day 10 (2021-01-11)
    dividend_date = datetime(2021, 1, 11)
    dividend = DividendAction(symbol="ASSET", date=dividend_date, amount=1.0)
    store = build_corporate_actions_store(dividends=[dividend])
    
    # Run backtest with corporate actions
    backtester = VectorBacktester()
    result_with_dividend = backtester.run(
        BuyAndHold,
        data={"ASSET": frame},
        capital=10000.0,
        corporate_actions=store,
    )
    
    # Run backtest without corporate actions (for comparison)
    result_without_dividend = backtester.run(
        BuyAndHold,
        data={"ASSET": frame},
        capital=10000.0,
    )
    
    # With 10000 capital and price 100, we should buy ~100 shares
    # Dividend of 1.0 per share on 100 shares = 100 cash added
    
    # Equity on day 10 (dividend date) should be higher with dividend
    equity_with_div_day10 = result_with_dividend.equity_series[10]
    equity_without_div_day10 = result_without_dividend.equity_series[10]
    
    # Difference should be approximately 100 (the dividend amount)
    difference = equity_with_div_day10 - equity_without_div_day10
    
    # Allow some tolerance for fees/slippage (should be small)
    assert difference >= 90, f"Dividend should increase equity by ~100, got {difference}"
    assert difference <= 110, f"Dividend should increase equity by ~100, got {difference}"
    
    # Final equity should also be higher
    final_equity_with = result_with_dividend.equity_series.iloc[-1]
    final_equity_without = result_without_dividend.equity_series.iloc[-1]
    
    assert final_equity_with > final_equity_without, "Final equity should be higher with dividends"
    
    # Both should have same number of equity points
    assert len(result_with_dividend.equity_series) == len(result_without_dividend.equity_series)


def test_portfolio_apply_split():
    """Test that Portfolio.apply_split correctly adjusts position size and cost basis."""
    portfolio = Portfolio(starting_cash=10000.0)
    
    # Create a position: 100 shares at $50 per share
    from alphaweave.core.types import Position, Fill
    from datetime import datetime
    
    fill = Fill(
        order_id=1,
        symbol="TEST",
        size=100.0,
        price=50.0,
        datetime=datetime(2021, 1, 1),
    )
    portfolio.apply_fill(fill)
    
    # Verify initial position
    position = portfolio.positions["TEST"]
    assert position.size == 100.0
    assert position.avg_price == 50.0
    assert portfolio.cash == 10000.0 - (100.0 * 50.0)  # 5000
    
    # Apply 2-for-1 split
    portfolio.apply_split("TEST", ratio=2.0)
    
    # Verify split results
    position = portfolio.positions["TEST"]
    assert position.size == 200.0, f"Expected 200 shares after 2-for-1 split, got {position.size}"
    assert position.avg_price == 25.0, f"Expected $25 avg price after split, got {position.avg_price}"
    
    # Total cost should be unchanged
    total_cost = position.size * position.avg_price
    assert abs(total_cost - 5000.0) < 1e-10, f"Total cost should be unchanged, got {total_cost}"
    
    # Cash should be unchanged
    assert portfolio.cash == 5000.0


def test_corporate_actions_store():
    """Test CorporateActionsStore functionality."""
    store = CorporateActionsStore()
    
    # Add splits
    split1 = SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0)
    split2 = SplitAction(symbol="AAPL", date=datetime(2023, 1, 1), ratio=2.0)
    store.add_split(split1)
    store.add_split(split2)
    
    # Add dividends
    div1 = DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24)
    div2 = DividendAction(symbol="AAPL", date=datetime(2023, 12, 15), amount=0.24)
    store.add_dividend(div1)
    store.add_dividend(div2)
    
    # Test retrieval
    splits_on_date = store.get_splits_on_date("AAPL", datetime(2020, 8, 31))
    assert len(splits_on_date) == 1
    assert splits_on_date[0].ratio == 4.0
    
    dividends_on_date = store.get_dividends_on_date("AAPL", datetime(2023, 11, 16))
    assert len(dividends_on_date) == 1
    assert dividends_on_date[0].amount == 0.24
    
    # Test no actions for other dates
    assert len(store.get_splits_on_date("AAPL", datetime(2021, 1, 1))) == 0
    assert len(store.get_dividends_on_date("AAPL", datetime(2023, 1, 1))) == 0
    
    # Test has_actions_for_symbol
    assert store.has_actions_for_symbol("AAPL") is True
    assert store.has_actions_for_symbol("MSFT") is False


def test_build_corporate_actions_store():
    """Test build_corporate_actions_store helper function."""
    splits = [
        SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0),
        SplitAction(symbol="MSFT", date=datetime(2003, 2, 18), ratio=2.0),
    ]
    
    dividends = [
        DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24),
        DividendAction(symbol="MSFT", date=datetime(2023, 11, 15), amount=0.75),
    ]
    
    store = build_corporate_actions_store(splits=splits, dividends=dividends)
    
    assert store.has_actions_for_symbol("AAPL") is True
    assert store.has_actions_for_symbol("MSFT") is True
    
    aapl_splits = store.get_splits_on_date("AAPL", datetime(2020, 8, 31))
    assert len(aapl_splits) == 1
    assert aapl_splits[0].ratio == 4.0
    
    msft_dividends = store.get_dividends_on_date("MSFT", datetime(2023, 11, 15))
    assert len(msft_dividends) == 1
    assert msft_dividends[0].amount == 0.75

