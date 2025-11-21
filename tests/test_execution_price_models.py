"""Tests for execution price models."""

from datetime import datetime

from alphaweave.core.types import Bar, Order, OrderType
from alphaweave.execution.price_models import (
    MidpointPriceModel,
    VWAPPriceModel,
    OpenClosePriceModel,
    ClosePriceModel,
)


def test_midpoint_price_model():
    """Test MidpointPriceModel."""
    model = MidpointPriceModel()
    bar = Bar(
        datetime=datetime(2021, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=103.0,
        volume=1000.0,
    )
    order = Order(symbol="TEST", size=100.0, order_type=OrderType.MARKET)
    
    price = model.get_fill_price(bar, order)
    assert price == 100.0  # (105 + 95) / 2


def test_vwap_price_model():
    """Test VWAPPriceModel."""
    model = VWAPPriceModel()
    bar = Bar(
        datetime=datetime(2021, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=103.0,
        volume=1000.0,
    )
    order = Order(symbol="TEST", size=100.0, order_type=OrderType.MARKET)
    
    price = model.get_fill_price(bar, order)
    assert price == 100.75  # (100 + 105 + 95 + 103) / 4


def test_open_close_price_model():
    """Test OpenClosePriceModel."""
    model_open = OpenClosePriceModel(use_open=True)
    model_close = OpenClosePriceModel(use_open=False)
    
    bar = Bar(
        datetime=datetime(2021, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=103.0,
        volume=1000.0,
    )
    order = Order(symbol="TEST", size=100.0, order_type=OrderType.MARKET)
    
    price_open = model_open.get_fill_price(bar, order)
    price_close = model_close.get_fill_price(bar, order)
    
    assert price_open == 100.0
    assert price_close == 103.0


def test_close_price_model():
    """Test ClosePriceModel (backward compatibility)."""
    model = ClosePriceModel()
    bar = Bar(
        datetime=datetime(2021, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=103.0,
        volume=1000.0,
    )
    order = Order(symbol="TEST", size=100.0, order_type=OrderType.MARKET)
    
    price = model.get_fill_price(bar, order)
    assert price == 103.0

