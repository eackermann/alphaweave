"""Execution price models for intraday execution realism."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from alphaweave.core.types import Bar, Order


class ExecutionPriceModel(Protocol):
    """Protocol for execution price models."""

    def get_fill_price(self, bar: Bar, order: Order) -> float:
        """
        Return the execution price for the given order in this bar.

        Args:
            bar: The bar data (OHLCV)
            order: The order to execute

        Returns:
            Execution price for the order
        """
        ...


class MidpointPriceModel:
    """
    Execution price model using bar midpoint.

    Uses (high + low) / 2, or (bid + ask) / 2 if bid/ask available.
    Falls back to (open + close) / 2 if high/low not available.
    """

    def get_fill_price(self, bar: Bar, order: Order) -> float:
        """Return midpoint price."""
        # If bid/ask available (future enhancement)
        # For now, use (high + low) / 2
        if bar.high is not None and bar.low is not None:
            return (bar.high + bar.low) / 2.0
        # Fallback to (open + close) / 2
        return (bar.open + bar.close) / 2.0


class VWAPPriceModel:
    """
    Execution price model using Volume-Weighted Average Price.

    Uses VWAP column if available, else approximates via (open + high + low + close) / 4.
    """

    def __init__(self, use_vwap_column: bool = True):
        """
        Initialize VWAP price model.

        Args:
            use_vwap_column: If True, look for 'vwap' column in bar data
        """
        self.use_vwap_column = use_vwap_column

    def get_fill_price(self, bar: Bar, order: Order) -> float:
        """
        Return VWAP price.

        Note: Bar dataclass doesn't have vwap field, so we approximate.
        In practice, this would be called with a bar-like object that may have vwap.
        """
        # For now, approximate VWAP as (open + high + low + close) / 4
        # This is a common approximation when true VWAP isn't available
        return (bar.open + bar.high + bar.low + bar.close) / 4.0


class OpenClosePriceModel:
    """
    Execution price model using open or close price.

    Useful for "trade at next open" semantics.
    """

    def __init__(self, use_open: bool = True):
        """
        Initialize open/close price model.

        Args:
            use_open: If True, use open price; if False, use close price
        """
        self.use_open = use_open

    def get_fill_price(self, bar: Bar, order: Order) -> float:
        """Return open or close price based on configuration."""
        return bar.open if self.use_open else bar.close


class ClosePriceModel:
    """
    Execution price model using close price.

    This is the default behavior for backward compatibility.
    """

    def get_fill_price(self, bar: Bar, order: Order) -> float:
        """Return close price."""
        return bar.close

