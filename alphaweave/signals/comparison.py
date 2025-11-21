"""Comparison signal implementations."""

from typing import Any, Union
from alphaweave.signals.base import Signal
from alphaweave.indicators.base import Indicator


class GreaterThan(Signal):
    """Signal triggered when indicator value is greater than threshold."""

    def __init__(self, indicator: Indicator, threshold: Union[float, Indicator]):
        """
        Initialize GreaterThan signal.

        Args:
            indicator: Indicator to check
            threshold: Threshold value or another indicator
        """
        self.indicator = indicator
        self.threshold = threshold

    def __call__(self, index: Any) -> bool:
        """
        Check if indicator > threshold at index.

        Args:
            index: Bar index

        Returns:
            True if indicator value > threshold
        """
        try:
            val = self.indicator[index]
            if isinstance(self.threshold, Indicator):
                threshold_val = self.threshold[index]
            else:
                threshold_val = self.threshold
            
            return val > threshold_val
        except (IndexError, KeyError, ValueError):
            return False


class LessThan(Signal):
    """Signal triggered when indicator value is less than threshold."""

    def __init__(self, indicator: Indicator, threshold: Union[float, Indicator]):
        """
        Initialize LessThan signal.

        Args:
            indicator: Indicator to check
            threshold: Threshold value or another indicator
        """
        self.indicator = indicator
        self.threshold = threshold

    def __call__(self, index: Any) -> bool:
        """
        Check if indicator < threshold at index.

        Args:
            index: Bar index

        Returns:
            True if indicator value < threshold
        """
        try:
            val = self.indicator[index]
            if isinstance(self.threshold, Indicator):
                threshold_val = self.threshold[index]
            else:
                threshold_val = self.threshold
            
            return val < threshold_val
        except (IndexError, KeyError, ValueError):
            return False

