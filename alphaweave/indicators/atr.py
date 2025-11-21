"""Average True Range (ATR) indicator."""

import pandas as pd
from alphaweave.indicators.base import Indicator
from alphaweave.core.frame import Frame


class ATR(Indicator):
    """Average True Range indicator."""

    def __init__(
        self,
        source: Frame,
        period: int = 14,
    ):
        """
        Initialize ATR indicator.

        Args:
            source: Frame containing the data
            period: ATR period (default: 14)
        """
        super().__init__(source, column="close")  # column not used for ATR
        if period < 1:
            raise ValueError("Period must be >= 1")
        self.period = period

    def _compute_all(self) -> None:
        """Override to compute ATR from full DataFrame (needs high, low, close)."""
        # ATR requires a Frame with high, low, close columns
        if not isinstance(self.source, Frame):
            raise ValueError("ATR requires a Frame source with 'high', 'low', and 'close' columns")
        
        pdf = self.source.to_pandas()
        
        # ATR requires high, low, close
        if "high" not in pdf.columns or "low" not in pdf.columns or "close" not in pdf.columns:
            raise ValueError("ATR requires 'high', 'low', and 'close' columns")
        
        high = pdf["high"]
        low = pdf["low"]
        close = pdf["close"]
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = tr.rolling(window=self.period, min_periods=1).mean()
        
        self._values = atr

    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute ATR values (not used, ATR overrides _compute_all instead).

        Args:
            series: Input price series (not used)

        Returns:
            Series with ATR values
        """
        # This method is not used for ATR, but required by base class
        # ATR overrides _compute_all() instead
        return pd.Series(dtype=float)

