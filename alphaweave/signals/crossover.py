"""Crossover signal implementations."""

from typing import Any
from alphaweave.signals.base import Signal
from alphaweave.indicators.base import Indicator


class CrossOver(Signal):
    """Signal triggered when first indicator crosses above second indicator."""

    def __init__(self, indicator1: Indicator, indicator2: Indicator):
        """
        Initialize CrossOver signal.

        Args:
            indicator1: First indicator (must cross above)
            indicator2: Second indicator (must cross below)
        """
        self.indicator1 = indicator1
        self.indicator2 = indicator2

    def __call__(self, index: Any) -> bool:
        """
        Check if crossover occurred at index.

        Args:
            index: Bar index

        Returns:
            True if indicator1 crossed above indicator2 at this bar
        """
        if index == 0:
            return False  # Need previous bar for comparison
        
        try:
            val1_current = self.indicator1[index]
            val2_current = self.indicator2[index]
            val1_prev = self.indicator1[index - 1]
            val2_prev = self.indicator2[index - 1]
            
            # Crossover: indicator1 was below indicator2, now above
            return val1_prev < val2_prev and val1_current > val2_current
        except (IndexError, KeyError):
            return False


class CrossUnder(Signal):
    """Signal triggered when first indicator crosses below second indicator."""

    def __init__(self, indicator1: Indicator, indicator2: Indicator):
        """
        Initialize CrossUnder signal.

        Args:
            indicator1: First indicator (must cross below)
            indicator2: Second indicator (must cross above)
        """
        self.indicator1 = indicator1
        self.indicator2 = indicator2

    def __call__(self, index: Any) -> bool:
        """
        Check if crossunder occurred at index.

        Args:
            index: Bar index

        Returns:
            True if indicator1 crossed below indicator2 at this bar
        """
        if index == 0:
            return False  # Need previous bar for comparison
        
        try:
            val1_current = self.indicator1[index]
            val2_current = self.indicator2[index]
            val1_prev = self.indicator1[index - 1]
            val2_prev = self.indicator2[index - 1]
            
            # Crossunder: indicator1 was above indicator2, now below
            return val1_prev > val2_prev and val1_current < val2_current
        except (IndexError, KeyError):
            return False

