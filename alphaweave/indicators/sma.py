"""Simple Moving Average (SMA) indicator."""

from typing import Optional
import pandas as pd
from alphaweave.indicators.base import Indicator


class SMA(Indicator):
    """Simple Moving Average indicator."""

    def __init__(
        self,
        source: Frame,
        period: int,
        column: str = "close",
    ):
        """
        Initialize SMA indicator.

        Args:
            source: Frame containing the data
            period: Moving average period
            column: Column name to use (default: "close")
        """
        super().__init__(source, column)
        if period < 1:
            raise ValueError("Period must be >= 1")
        self.period = period

    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute SMA values.

        Args:
            series: Input price series

        Returns:
            Series with SMA values
        """
        return series.rolling(window=self.period, min_periods=1).mean()

