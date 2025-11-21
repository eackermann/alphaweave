"""Relative Strength Index (RSI) indicator."""

import pandas as pd
import numpy as np
from alphaweave.indicators.base import Indicator


class RSI(Indicator):
    """Relative Strength Index indicator."""

    def __init__(
        self,
        source: Frame,
        period: int = 14,
        column: str = "close",
    ):
        """
        Initialize RSI indicator.

        Args:
            source: Frame containing the data
            period: RSI period (default: 14)
            column: Column name to use (default: "close")
        """
        super().__init__(source, column)
        if period < 1:
            raise ValueError("Period must be >= 1")
        self.period = period

    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute RSI values.

        Args:
            series: Input price series

        Returns:
            Series with RSI values (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period, min_periods=1).mean()
        
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Default to 50 when no data

