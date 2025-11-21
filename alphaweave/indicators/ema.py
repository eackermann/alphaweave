"""Exponential Moving Average (EMA) indicator."""

from typing import Optional
import pandas as pd
from alphaweave.indicators.base import Indicator


class EMA(Indicator):
    """Exponential Moving Average indicator."""

    def __init__(
        self,
        source: Frame,
        period: int,
        column: str = "close",
    ):
        """
        Initialize EMA indicator.

        Args:
            source: Frame containing the data
            period: EMA period
            column: Column name to use (default: "close")
        """
        super().__init__(source, column)
        if period < 1:
            raise ValueError("Period must be >= 1")
        self.period = period

    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute EMA values.

        Args:
            series: Input price series

        Returns:
            Series with EMA values
        """
        return series.ewm(span=self.period, adjust=False).mean()

