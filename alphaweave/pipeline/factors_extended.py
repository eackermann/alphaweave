"""Extended factor library for Sprint 13."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.pipeline.factors import Factor, VolatilityFactor, _align_dataframes


# ============================================================================
# Momentum Variants
# ============================================================================

class TimeSeriesMomentumFactor(Factor):
    """Time-series momentum factor (longer lookback periods)."""

    def __init__(self, period: int = 252, field: str = "close", name: Optional[str] = None):
        """
        Initialize time-series momentum factor.

        Args:
            period: Lookback window in periods (default: 252 for 1 year)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute time-series momentum for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            momentum = series.pct_change(periods=self.period)
            result[symbol] = momentum

        return _align_dataframes(result)


class CrossSectionalMomentumFactor(Factor):
    """Cross-sectional momentum (rank-based)."""

    def __init__(self, period: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize cross-sectional momentum factor.

        Args:
            period: Lookback window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.period = period
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute cross-sectional momentum (returns, will be ranked later)."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            momentum = series.pct_change(periods=self.period)
            result[symbol] = momentum

        return _align_dataframes(result)


# ============================================================================
# Volatility Estimators
# ============================================================================

class GarmanKlassVolFactor(Factor):
    """Garman-Klass volatility estimator using OHLC data."""

    def __init__(self, window: int = 63, name: Optional[str] = None):
        """
        Initialize Garman-Klass volatility factor.

        Args:
            window: Rolling window in periods (default: 63)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute Garman-Klass volatility for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            required = ["open", "high", "low", "close"]
            if not all(col in df.columns for col in required):
                continue

            # Garman-Klass estimator
            log_hl = np.log(df["high"] / df["low"])
            log_co = np.log(df["close"] / df["open"])
            gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            vol = np.sqrt(gk.rolling(window=self.window, min_periods=1).mean()) * np.sqrt(252)
            result[symbol] = vol

        return _align_dataframes(result)


class ParkinsonVolFactor(Factor):
    """Parkinson volatility estimator using high-low data."""

    def __init__(self, window: int = 63, name: Optional[str] = None):
        """
        Initialize Parkinson volatility factor.

        Args:
            window: Rolling window in periods (default: 63)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute Parkinson volatility for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "high" not in df.columns or "low" not in df.columns:
                continue

            # Parkinson estimator
            log_hl = np.log(df["high"] / df["low"])
            parkinson = (1 / (4 * np.log(2))) * log_hl**2
            vol = np.sqrt(parkinson.rolling(window=self.window, min_periods=1).mean()) * np.sqrt(252)
            result[symbol] = vol

        return _align_dataframes(result)


class ATRVolFactor(Factor):
    """ATR-based volatility (ATR normalized by price)."""

    def __init__(self, window: int = 14, field: str = "close", name: Optional[str] = None):
        """
        Initialize ATR volatility factor.

        Args:
            window: ATR window in periods (default: 14)
            field: Field to use for normalization (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute ATR-based volatility for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            required = ["high", "low", "close"]
            if not all(col in df.columns for col in required) or self.field not in df.columns:
                continue

            # True Range
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR
            atr = tr.rolling(window=self.window, min_periods=1).mean()

            # Normalize by price
            atr_vol = (atr / df[self.field]) * np.sqrt(252)
            result[symbol] = atr_vol

        return _align_dataframes(result)


# ============================================================================
# Higher Moments
# ============================================================================

class ReturnSkewFactor(Factor):
    """Rolling skewness of returns."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize return skewness factor.

        Args:
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute return skewness for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            returns = series.pct_change()
            skew = returns.rolling(window=self.window, min_periods=1).skew()
            result[symbol] = skew

        return _align_dataframes(result)


class ReturnKurtosisFactor(Factor):
    """Rolling kurtosis of returns."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize return kurtosis factor.

        Args:
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute return kurtosis for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue
            series = df[self.field]
            returns = series.pct_change()
            kurt = returns.rolling(window=self.window, min_periods=1).kurt()
            result[symbol] = kurt

        return _align_dataframes(result)


# ============================================================================
# Trend Factors
# ============================================================================

class TrendSlopeFactor(Factor):
    """Trend slope (regression slope of log price)."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize trend slope factor.

        Args:
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute trend slope for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            log_price = np.log(df[self.field])
            slopes = []
            for i in range(len(log_price)):
                window_data = log_price.iloc[max(0, i - self.window + 1) : i + 1]
                if len(window_data) < 2:
                    slopes.append(np.nan)
                    continue

                # Linear regression slope
                x = np.arange(len(window_data))
                slope = np.polyfit(x, window_data.values, 1)[0]
                slopes.append(slope)

            result[symbol] = pd.Series(slopes, index=log_price.index)

        return _align_dataframes(result)


class TrendStrengthFactor(Factor):
    """Trend strength (R² of rolling regression)."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize trend strength factor.

        Args:
            window: Rolling window in periods (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute trend strength (R²) for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            log_price = np.log(df[self.field])
            r_squared = []
            for i in range(len(log_price)):
                window_data = log_price.iloc[max(0, i - self.window + 1) : i + 1]
                if len(window_data) < 2:
                    r_squared.append(np.nan)
                    continue

                # Linear regression R²
                x = np.arange(len(window_data))
                coeffs = np.polyfit(x, window_data.values, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((window_data.values - y_pred) ** 2)
                ss_tot = np.sum((window_data.values - np.mean(window_data.values)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                r_squared.append(r2)

            result[symbol] = pd.Series(r_squared, index=log_price.index)

        return _align_dataframes(result)


# ============================================================================
# Style & Risk Factors
# ============================================================================

class LowVolatilityFactor(Factor):
    """Low volatility score (inverse rank of volatility)."""

    def __init__(self, window: int = 63, field: str = "close", name: Optional[str] = None):
        """
        Initialize low volatility factor.

        Args:
            window: Rolling window for volatility calculation (default: 63)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute low volatility score (will be ranked cross-sectionally)."""
        # Compute volatility first
        vol_factor = VolatilityFactor(window=self.window, field=self.field)
        vol_df = vol_factor.compute(data)
        # Return negative volatility (lower vol = higher score)
        return -vol_df


class TurnoverFactor(Factor):
    """Turnover factor (volume / shares outstanding, or volume proxy)."""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        """
        Initialize turnover factor.

        Args:
            window: Rolling window for average (default: 20)
            name: Optional name for the factor
        """
        super().__init__(name)
        self.window = window

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute turnover for all symbols."""
        result = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if "volume" not in df.columns:
                continue

            # Simple turnover proxy: average volume
            turnover = df["volume"].rolling(window=self.window, min_periods=1).mean()
            result[symbol] = turnover

        return _align_dataframes(result)


class IdiosyncraticVolFactor(Factor):
    """Idiosyncratic volatility (residual vol from beta regression)."""

    def __init__(
        self,
        benchmark: str,
        window: int = 252,
        field: str = "close",
        name: Optional[str] = None,
    ):
        """
        Initialize idiosyncratic volatility factor.

        Args:
            benchmark: Symbol name of the benchmark
            window: Rolling window in periods (default: 252)
            field: Field to use (default: "close")
            name: Optional name for the factor
        """
        super().__init__(name)
        self.benchmark = benchmark
        self.window = window
        self.field = field

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute idiosyncratic volatility for all symbols."""
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
                # Idio vol of benchmark is 0
                df = frame.to_pandas()
                if self.field in df.columns:
                    result[symbol] = pd.Series(0.0, index=df.index)
                continue

            df = frame.to_pandas()
            if self.field not in df.columns:
                continue

            asset_returns = df[self.field].pct_change()

            # Align returns
            aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()

            if len(aligned) < self.window:
                continue

            # Rolling regression to get residuals
            idio_vols = []
            for i in range(len(aligned)):
                window_data = aligned.iloc[max(0, i - self.window + 1) : i + 1]
                if len(window_data) < 2:
                    idio_vols.append(np.nan)
                    continue

                # OLS regression
                x = window_data["bench"].values
                y = window_data["asset"].values
                if np.var(x) == 0:
                    idio_vols.append(np.nan)
                    continue

                beta = np.cov(y, x)[0, 1] / np.var(x)
                alpha = np.mean(y) - beta * np.mean(x)
                residuals = y - (alpha + beta * x)
                idio_vol = np.std(residuals) * np.sqrt(252)  # Annualized
                idio_vols.append(idio_vol)

            result[symbol] = pd.Series(idio_vols, index=aligned.index)

        return _align_dataframes(result)


# ============================================================================
# Size Factor (Placeholder - requires market cap data)
# ============================================================================

class LogMarketCapFactor(Factor):
    """Log market capitalization factor (requires market cap input)."""

    def __init__(self, market_cap_data: Optional[Dict[str, pd.DataFrame]] = None, name: Optional[str] = None):
        """
        Initialize log market cap factor.

        Args:
            market_cap_data: Optional dict of symbol -> DataFrame with market cap values.
                           If None, will use price * volume as proxy.
            name: Optional name for the factor
        """
        super().__init__(name)
        self.market_cap_data = market_cap_data or {}

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute log market cap for all symbols."""
        result = {}
        for symbol, frame in data.items():
            if symbol in self.market_cap_data:
                # Use provided market cap data
                mcap_df = self.market_cap_data[symbol]
                log_mcap = np.log(mcap_df.iloc[:, 0])  # Use first column
                result[symbol] = log_mcap
            else:
                # Proxy: log(price * volume)
                df = frame.to_pandas()
                if "close" in df.columns and "volume" in df.columns:
                    proxy_mcap = df["close"] * df["volume"]
                    log_mcap = np.log(proxy_mcap.replace(0, np.nan))
                    result[symbol] = log_mcap

        return _align_dataframes(result)


# ============================================================================
# Fundamental Factor Placeholders
# ============================================================================

class BookToPriceFactor(Factor):
    """Book-to-price ratio factor (requires fundamental data)."""

    def __init__(self, book_value_data: Optional[Dict[str, pd.DataFrame]] = None, name: Optional[str] = None):
        """
        Initialize book-to-price factor.

        Args:
            book_value_data: Dict of symbol -> DataFrame with book value per share
            name: Optional name for the factor
        """
        super().__init__(name)
        self.book_value_data = book_value_data or {}

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute book-to-price for all symbols."""
        result = {}
        for symbol, frame in data.items():
            if symbol not in self.book_value_data:
                continue

            df = frame.to_pandas()
            if "close" not in df.columns:
                continue

            book_df = self.book_value_data[symbol]
            # Align indices
            aligned = df["close"].align(book_df.iloc[:, 0], join="inner")
            btp = aligned[1] / aligned[0]
            result[symbol] = btp

        return _align_dataframes(result)


class EarningsToPriceFactor(Factor):
    """Earnings-to-price ratio factor (requires fundamental data)."""

    def __init__(self, earnings_data: Optional[Dict[str, pd.DataFrame]] = None, name: Optional[str] = None):
        """
        Initialize earnings-to-price factor.

        Args:
            earnings_data: Dict of symbol -> DataFrame with earnings per share
            name: Optional name for the factor
        """
        super().__init__(name)
        self.earnings_data = earnings_data or {}

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute earnings-to-price for all symbols."""
        result = {}
        for symbol, frame in data.items():
            if symbol not in self.earnings_data:
                continue

            df = frame.to_pandas()
            if "close" not in df.columns:
                continue

            earnings_df = self.earnings_data[symbol]
            # Align indices
            aligned = df["close"].align(earnings_df.iloc[:, 0], join="inner")
            etp = aligned[1] / aligned[0]
            result[symbol] = etp

        return _align_dataframes(result)


class DividendYieldFactor(Factor):
    """Dividend yield factor (requires dividend data)."""

    def __init__(self, dividend_data: Optional[Dict[str, pd.DataFrame]] = None, name: Optional[str] = None):
        """
        Initialize dividend yield factor.

        Args:
            dividend_data: Dict of symbol -> DataFrame with dividend per share
            name: Optional name for the factor
        """
        super().__init__(name)
        self.dividend_data = dividend_data or {}

    def compute(self, data: Dict[str, Frame]) -> pd.DataFrame:
        """Compute dividend yield for all symbols."""
        result = {}
        for symbol, frame in data.items():
            if symbol not in self.dividend_data:
                continue

            df = frame.to_pandas()
            if "close" not in df.columns:
                continue

            div_df = self.dividend_data[symbol]
            # Align indices
            aligned = df["close"].align(div_df.iloc[:, 0], join="inner")
            yield_val = aligned[1] / aligned[0]
            result[symbol] = yield_val

        return _align_dataframes(result)

