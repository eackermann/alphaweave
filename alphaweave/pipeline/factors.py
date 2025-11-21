"""Factor computation for cross-sectional and time-series analysis."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.pipeline.expressions import FactorExpression


class Factor(ABC):
    """Base class for factors that compute values across symbols and time."""

    def __init__(self, name: Optional[str] = None):
        """
        Initialize factor.

        Args:
            name: Optional name for the factor (for debugging/caching)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """
        Compute factor values for all symbols.

        Args:
            data: Dictionary mapping symbol names to Frame objects

        Returns:
            DataFrame with datetime index and symbol columns.
            Shape: [datetime × symbols]
        """
        raise NotImplementedError

    def zscore(self, window: Optional[int] = None) -> "FactorExpression":
        """
        Create a z-score normalized version of this factor.

        Args:
            window: Optional rolling window for time-series z-score.
                   If None, computes cross-sectional z-score per bar.

        Returns:
            FactorExpression that applies z-score normalization
        """
        return FactorExpression(self).zscore(window=window)

    def rank(self, ascending: bool = False) -> "FactorExpression":
        """
        Create a ranked version of this factor.

        Args:
            ascending: If True, rank ascending (lowest=1). If False, rank descending (highest=1).

        Returns:
            FactorExpression that applies ranking
        """
        return FactorExpression(self).rank(ascending=ascending)

    def percentile(self) -> "FactorExpression":
        """
        Create a percentile-ranked version of this factor (0-100).

        Returns:
            FactorExpression that applies percentile ranking
        """
        return FactorExpression(self).percentile()

    def __add__(self, other):
        """Add two factors."""
        return FactorExpression(self) + other

    def __sub__(self, other):
        """Subtract two factors."""
        return FactorExpression(self) - other

    def __mul__(self, other):
        """Multiply two factors."""
        return FactorExpression(self) * other

    def __truediv__(self, other):
        """Divide two factors."""
        return FactorExpression(self) / other

    def __gt__(self, other):
        """Compare factor > other."""
        return FactorExpression(self) > other

    def __lt__(self, other):
        """Compare factor < other."""
        return FactorExpression(self) < other

    def __ge__(self, other):
        """Compare factor >= other."""
        return FactorExpression(self) >= other

    def __le__(self, other):
        """Compare factor <= other."""
        return FactorExpression(self) <= other


def _align_dataframes(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple DataFrames to a common datetime index.

    Args:
        frames: Dictionary of symbol -> DataFrame

    Returns:
        Aligned DataFrame with MultiIndex columns (symbol, field)
    """
    if not frames:
        return pd.DataFrame()

    # Get all datetime indices
    all_indices = set()
    for df in frames.values():
        all_indices.update(df.index)

    # Create master index (sorted)
    master_index = pd.DatetimeIndex(sorted(all_indices))

    # Align each DataFrame to master index
    aligned = {}
    for symbol, df in frames.items():
        aligned[symbol] = df.reindex(master_index)

    return pd.DataFrame(aligned)


class ReturnsFactor(Factor):
    """Compute returns over a specified window."""

    def __init__(self, window: int = 1, field: str = "close", name: Optional[str] = None):
        """
        Initialize returns factor.

        Args:
            window: Number of periods for return calculation (default: 1)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute returns for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            returns = series.pct_change(periods=self.window)
            result[symbol] = returns

        return _align_dataframes(result)


class MomentumFactor(Factor):
    """Compute momentum (lookback return) over a specified window."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize momentum factor.

        Args:
            window: Lookback window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute momentum for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            momentum = series.pct_change(periods=self.window)
            result[symbol] = momentum

        return _align_dataframes(result)


class VolatilityFactor(Factor):
    """Compute realized volatility (rolling std of returns)."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize volatility factor.

        Args:
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute volatility for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            returns = series.pct_change()
            vol = returns.rolling(window=self.window, min_periods=1).std() * np.sqrt(252)  # Annualized
            result[symbol] = vol

        return _align_dataframes(result)


class BetaFactor(Factor):
    """Compute rolling beta to a benchmark."""

    def __init__(
        self,
        benchmark: str,
        window: int = 63,
        field: str = "close",
        name: Optional[str] = None,
    ):
        """
        Initialize beta factor.

        Args:
            benchmark: Symbol name of the benchmark
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.benchmark = benchmark
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute beta for all symbols relative to benchmark."""
        if self.benchmark not in data:
            return pd.DataFrame()

        # Get benchmark returns
        bench_frame = data[self.benchmark]
        bench_df = bench_frame.to_pandas()
        if self.field not in bench_df.columns:
            return pd.DataFrame()

        bench_returns = bench_df[self.field].pct_change()

        result = {}
        for symbol, frame in data.items():
            if symbol == self.benchmark:
                # Beta to itself is 1.0
                df = frame.to_pandas()
                if self.field in df.columns:
                    result[symbol] = pd.Series(1.0, index=df.index)
                continue

            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            asset_returns = df[self.field].pct_change()

            # Align returns
            aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()

            if len(aligned) < self.window:
                continue

            # Rolling beta
            betas = []
            for i in range(len(aligned)):
                window_data = aligned.iloc[max(0, i - self.window + 1) : i + 1]
                if len(window_data) < 2:
                    betas.append(np.nan)
                    continue

                cov = window_data["asset"].cov(window_data["bench"])
                var = window_data["bench"].var()
                if var == 0 or np.isnan(var):
                    betas.append(np.nan)
                else:
                    betas.append(cov / var)

            result[symbol] = pd.Series(betas, index=aligned.index)

        return _align_dataframes(result)


class DollarVolumeFactor(Factor):
    """Compute average dollar volume (price * volume)."""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        """
        Initialize dollar volume factor.

        Args:
            window: Rolling window for average (default: 20)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute dollar volume for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "close" not in df.columns or "volume" not in df.columns:
                continue

            dollar_vol = df["close"] * df["volume"]
            avg_dollar_vol = dollar_vol.rolling(window=self.window, min_periods=1).mean()
            result[symbol] = avg_dollar_vol

        return _align_dataframes(result)


class SMAIndicatorFactor(Factor):
    """Compute SMA ratio (price / SMA)."""

    def __init__(self, period: int = 20, field: str = "close", name: Optional[str] = None):
        """
        Initialize SMA indicator factor.

        Args:
            period: SMA period (default: 20)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute SMA ratio for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            series = df[self.field]
            sma = series.rolling(window=self.period, min_periods=1).mean()
            ratio = series / sma
            result[symbol] = ratio

        return _align_dataframes(result)


class RSIFactor(Factor):
    """Compute RSI (Relative Strength Index)."""

    def __init__(self, period: int = 14, field: str = "close", name: Optional[str] = None):
        """
        Initialize RSI factor.

        Args:
            period: RSI period (default: 14)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute RSI for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            series = df[self.field]
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period, min_periods=1).mean()

            rs = gain / loss.replace(0, np.nan)
            rs = rs.fillna(0)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)  # Default to 50 when no data

            result[symbol] = rsi

        return _align_dataframes(result)

