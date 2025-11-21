"""Tests for the Portfolio class."""

from datetime import datetime

from alphaweave.core.types import Fill
from alphaweave.engine.portfolio import Portfolio


def make_fill(symbol: str, size: float, price: float) -> Fill:
    return Fill(order_id=1, symbol=symbol, size=size, price=price, datetime=datetime(2020, 1, 1))


def test_portfolio_apply_fill_buy_and_sell():
    portfolio = Portfolio(starting_cash=1_000.0)

    buy_fill = make_fill("AAPL", size=10, price=10)
    portfolio.apply_fill(buy_fill)

    assert portfolio.cash == 1_000.0 - 100.0
    assert "AAPL" in portfolio.positions
    position = portfolio.positions["AAPL"]
    assert position.size == 10
    assert position.avg_price == 10

    buy_fill_2 = make_fill("AAPL", size=5, price=12)
    portfolio.apply_fill(buy_fill_2)
    position = portfolio.positions["AAPL"]
    expected_avg = (10 * 10 + 5 * 12) / 15
    assert abs(position.avg_price - expected_avg) < 1e-9
    assert position.size == 15

    sell_fill = make_fill("AAPL", size=-8, price=11)
    portfolio.apply_fill(sell_fill)

    assert abs(portfolio.cash - (1_000.0 - 100.0 - 60.0 + 88.0)) < 1e-9
    position = portfolio.positions["AAPL"]
    assert position.size == 7
    assert position.avg_price == expected_avg


def test_portfolio_total_value():
    portfolio = Portfolio(starting_cash=500.0)
    portfolio.apply_fill(make_fill("AAPL", size=5, price=10))

    prices = {"AAPL": 12.0}
    total_value = portfolio.total_value(prices)
    assert abs(total_value - (500.0 - 50.0 + 5 * 12.0)) < 1e-9
