"""Technical indicator factors for Sprint 13."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.pipeline.factors import Factor, _align_dataframes


class MACDFactor(Factor):
    """MACD (Moving Average Convergence Divergence) factor."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        field: str = "close",
        name: Optional[str] = None,
    ):
        """
        Initialize MACD factor.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute MACD signal line for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            series = df[self.field]
            ema_fast = series.ewm(span=self.fast, adjust=False).mean()
            ema_slow = series.ewm(span=self.slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            result[symbol] = signal_line

        return _align_dataframes(result)


class StochasticKFactor(Factor):
    """Stochastic %K factor."""

    def __init__(self, period: int = 14, name: Optional[str] = None):
        """
        Initialize Stochastic %K factor.

        Args:
            period: Lookback period (default: 14)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute Stochastic %K for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
                continue

            high_max = df["high"].rolling(window=self.period, min_periods=1).max()
            low_min = df["low"].rolling(window=self.period, min_periods=1).min()
            k = 100 * (df["close"] - low_min) / (high_max - low_min)
            k = k.replace([np.inf, -np.inf], np.nan)
            result[symbol] = k.fillna(50)

        return _align_dataframes(result)


class BollingerZScoreFactor(Factor):
    """Bollinger Band Z-Score factor."""

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        field: str = "close",
        name: Optional[str] = None,
    ):
        """
        Initialize Bollinger Z-Score factor.

        Args:
            window: Moving average window (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.num_std = num_std
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute Bollinger Z-Score for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            series = df[self.field]
            ma = series.rolling(window=self.window, min_periods=1).mean()
            std = series.rolling(window=self.window, min_periods=1).std()
            z_score = (series - ma) / (std * self.num_std)
            z_score = z_score.replace([np.inf, -np.inf], np.nan)
            result[symbol] = z_score.fillna(0)

        return _align_dataframes(result)


class CCIFactor(Factor):
    """Commodity Channel Index (CCI) factor."""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        """
        Initialize CCI factor.

        Args:
            window: Lookback period (default: 20)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute CCI for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
                continue

            tp = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = tp.rolling(window=self.window, min_periods=1).mean()
            mad = tp.rolling(window=self.window, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            cci = (tp - sma_tp) / (0.015 * mad)
            cci = cci.replace([np.inf, -np.inf], np.nan)
            result[symbol] = cci.fillna(0)

        return _align_dataframes(result)


class WilliamsRFactor(Factor):
    """Williams %R factor."""

    def __init__(self, period: int = 14, name: Optional[str] = None):
        """
        Initialize Williams %R factor.

        Args:
            period: Lookback period (default: 14)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute Williams %R for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
                continue

            high_max = df["high"].rolling(window=self.period, min_periods=1).max()
            low_min = df["low"].rolling(window=self.period, min_periods=1).min()
            wr = -100 * (high_max - df["close"]) / (high_max - low_min)
            wr = wr.replace([np.inf, -np.inf], np.nan)
            result[symbol] = wr.fillna(-50)

        return _align_dataframes(result)

