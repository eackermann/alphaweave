"""Rate of Change (ROC) indicator."""

import pandas as pd
from alphaweave.indicators.base import Indicator


class ROC(Indicator):
    """Rate of Change indicator."""

    def __init__(
        self,
        source: Frame,
        period: int = 10,
        column: str = "close",
    ):
        """
        Initialize ROC indicator.

        Args:
            source: Frame containing the data
            period: ROC period (default: 10)
            column: Column name to use (default: "close")
        """
        super().__init__(source, column)
        if period < 1:
            raise ValueError("Period must be >= 1")
        self.period = period

    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute ROC values.

        Args:
            series: Input price series

        Returns:
            Series with ROC values (percentage change)
        """
        return series.pct_change(periods=self.period) * 100

