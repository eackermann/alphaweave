"""Tests for fee and slippage models."""

from datetime import datetime
from types import SimpleNamespace

from alphaweave.core.types import Fill, Order, OrderType
from alphaweave.execution.fees import NoFees, PercentageFees, PerShareFees
from alphaweave.execution.slippage import FixedBpsSlippage, NoSlippage


def make_order(size: float) -> Order:
    return Order(symbol="AAPL", size=size, order_type=OrderType.MARKET)


def make_fill(size: float, price: float) -> Fill:
    return Fill(order_id=1, symbol="AAPL", size=size, price=price, datetime=datetime(2020, 1, 1))


def test_fee_models():
    order = make_order(10)
    fill = make_fill(10, 100)

    assert NoFees().calculate(order, fill) == 0.0
    assert PerShareFees(0.01).calculate(order, fill) == 0.1
    assert PercentageFees(0.001).calculate(order, fill) == 1.0


def test_slippage_models():
    order_buy = make_order(10)
    order_sell = make_order(-10)
    bar = SimpleNamespace(close=100.0)

    assert NoSlippage().execute(order_buy, bar) == 100.0

    model = FixedBpsSlippage(50)  # 5 bps
    buy_price = model.execute(order_buy, bar)
    sell_price = model.execute(order_sell, bar)

    assert buy_price > bar.close
    assert sell_price < bar.close
