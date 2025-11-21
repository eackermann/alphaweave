"""Strategy base class for alphaweave."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from alphaweave.core.frame import Frame
from alphaweave.core.types import Order, OrderType
from alphaweave.data.timeframes import resample_frame
from alphaweave.indicators.sma import SMA
from alphaweave.indicators.ema import EMA


class Strategy:
    """Base class for trading strategies."""

    def __init__(self, data: Union[Frame, Dict[str, Frame]]):
        """
        Initialize strategy with data.

        Args:
            data: Either a Frame or a mapping symbol->Frame; Strategy implementations
                  should fetch assets from this data object.
        """
        self.data = data
        self._orders = []
        self._current_index: Optional[Any] = None
        self._current_timestamp: Optional[pd.Timestamp] = None
        self._indicator_cache: Dict[tuple, Any] = {}  # Cache for indicators
        self._timeframes: Dict[str, Dict[str, Frame]] = {
            "base": self._normalize_data_dict(data)
        }

    @staticmethod
    def _normalize_data_dict(data: Union[Frame, Dict[str, Frame]]) -> Dict[str, Frame]:
        if isinstance(data, Frame):
            return {"_default": data}
        elif isinstance(data, dict):
            return dict(data)
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")

    def _get_timeframe_dict(self, timeframe: Optional[str]) -> Dict[str, Frame]:
        key = "base" if timeframe in (None, "base") else timeframe
        if key not in self._timeframes:
            raise ValueError(f"timeframe '{timeframe}' has not been registered")
        return self._timeframes[key]

    def _set_current_index(self, index: Any) -> None:
        """
        Set the current bar index. Called by the backtester before each bar.

        Args:
            index: Current bar index (integer or pandas.Timestamp)
        """
        self._current_index = index

    def _set_current_timestamp(self, ts: pd.Timestamp) -> None:
        """Set the current bar timestamp."""
        self._current_timestamp = ts

    def init(self) -> None:
        """Called once before backtest loop. Subclasses may override."""
        pass

    def next(self, index: Any) -> None:
        """
        Called once per bar. index is an integer or pandas.Timestamp index position.

        Args:
            index: Current bar index (integer or pandas.Timestamp)
        """
        raise NotImplementedError("Subclasses must implement next()")

    def register_timeframe(
        self, name: str, rule: str, symbols: Optional[List[str]] = None
    ) -> None:
        """
        Register a resampled timeframe derived from the base data.
        """
        if name == "base":
            raise ValueError("Cannot register timeframe with reserved name 'base'")

        base_frames = self._timeframes["base"]
        if symbols is None:
            symbols = list(base_frames.keys())

        tf_frames: Dict[str, Frame] = {}
        for symbol in symbols:
            if symbol not in base_frames:
                raise ValueError(f"Symbol '{symbol}' not found in base timeframe")
            tf_frames[symbol] = resample_frame(base_frames[symbol], rule)

        self._timeframes[name] = tf_frames

    def get_frame(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> Frame:
        """
        Get Frame for a symbol. If data is a single Frame, return it.
        If data is a dict, return the Frame for the given symbol.

        Args:
            symbol: Symbol name (required if data is a dict with multiple symbols)

        Returns:
            Frame for the symbol

        Raises:
            ValueError: If symbol is required but not provided, or symbol not found
        """
        frames = self._get_timeframe_dict(timeframe)
        if symbol is None:
            if len(frames) == 1:
                return next(iter(frames.values()))
            raise ValueError("symbol required when data contains multiple symbols")
        if symbol in frames:
            return frames[symbol]
        if len(frames) == 1:
            return next(iter(frames.values()))
        raise ValueError(f"Symbol '{symbol}' not found in timeframe '{timeframe or 'base'}'")

    def series(
        self,
        symbol: Optional[str] = None,
        field: str = "close",
        timeframe: Optional[str] = None,
    ) -> pd.Series:
        """
        Get a pandas Series for a symbol and field.

        Args:
            symbol: Symbol name (required if data is a dict with multiple symbols)
            field: Field name (default: "close")

        Returns:
            pandas Series with the field data
        """
        frame = self.get_frame(symbol, timeframe=timeframe)
        pdf = frame.to_pandas()
        if field not in pdf.columns:
            raise ValueError(f"Field '{field}' not found in Frame")
        return pdf[field]

    def close(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> float:
        """
        Get the close price for the current bar.

        Args:
            symbol: Symbol name (required if data is a dict with multiple symbols)

        Returns:
            Close price for the current bar

        Raises:
            ValueError: If _current_index is not set
        """
        frame = self.get_frame(symbol, timeframe=timeframe)
        pdf = frame.to_pandas()

        if timeframe in (None, "base"):
            if self._current_index is None:
                raise ValueError(
                    "_current_index not set. This method should be called from next()"
                )
            if isinstance(self._current_index, int):
                return float(pdf.iloc[self._current_index]["close"])
            else:
                return float(pdf.loc[self._current_index, "close"])

        ts = self.now()
        if ts is None:
            raise ValueError("Current timestamp is not available for timeframe lookup")
        ts = pd.Timestamp(ts)
        sliced = pdf.loc[:ts]
        if sliced.empty:
            raise ValueError(f"No data available for timeframe '{timeframe}' at {ts}")
        return float(sliced.iloc[-1]["close"])

    def sma(self, symbol: Optional[str] = None, period: int = 20, field: str = "close") -> float:
        """
        Get Simple Moving Average value for the current bar.

        Args:
            symbol: Symbol name (required if data is a dict with multiple symbols)
            period: SMA period (default: 20)
            field: Field name (default: "close")

        Returns:
            SMA value for the current bar

        Raises:
            ValueError: If _current_index is not set
        """
        if self._current_index is None:
            raise ValueError("_current_index not set. This method should be called from next()")
        
        # Cache key: (indicator_kind, symbol, period, field)
        cache_key = ("sma", symbol, period, field)
        
        if cache_key not in self._indicator_cache:
            series = self.series(symbol, field)
            self._indicator_cache[cache_key] = SMA(series, period=period)
        
        indicator = self._indicator_cache[cache_key]
        return indicator[self._current_index]

    def ema(self, symbol: Optional[str] = None, period: int = 20, field: str = "close") -> float:
        """
        Get Exponential Moving Average value for the current bar.

        Args:
            symbol: Symbol name (required if data is a dict with multiple symbols)
            period: EMA period (default: 20)
            field: Field name (default: "close")

        Returns:
            EMA value for the current bar

        Raises:
            ValueError: If _current_index is not set
        """
        if self._current_index is None:
            raise ValueError("_current_index not set. This method should be called from next()")
        
        # Cache key: (indicator_kind, symbol, period, field)
        cache_key = ("ema", symbol, period, field)
        
        if cache_key not in self._indicator_cache:
            series = self.series(symbol, field)
            self._indicator_cache[cache_key] = EMA(series, period=period)
        
        indicator = self._indicator_cache[cache_key]
        return indicator[self._current_index]

    def order_market(self, symbol: str, size: float) -> None:
        """Submit an IOC market order."""
        self._orders.append(
            Order(symbol=symbol, size=float(size), order_type=OrderType.MARKET)
        )

    def order_limit(self, symbol: str, size: float, limit_price: float) -> None:
        """Submit a limit order."""
        self._orders.append(
            Order(
                symbol=symbol,
                size=float(size),
                order_type=OrderType.LIMIT,
                limit_price=float(limit_price),
            )
        )

    def order_stop(self, symbol: str, size: float, stop_price: float) -> None:
        """Submit a stop order."""
        self._orders.append(
            Order(
                symbol=symbol,
                size=float(size),
                order_type=OrderType.STOP,
                stop_price=float(stop_price),
            )
        )

    def order_stop_limit(
        self,
        symbol: str,
        size: float,
        stop_price: float,
        limit_price: float,
    ) -> None:
        """Submit a stop-limit order."""
        self._orders.append(
            Order(
                symbol=symbol,
                size=float(size),
                order_type=OrderType.STOP_LIMIT,
                stop_price=float(stop_price),
                limit_price=float(limit_price),
            )
        )

    def order_target_percent(self, symbol: str, target: float) -> None:
        """
        Request to set target percent of portfolio in `symbol`.

        Args:
            symbol: Symbol to target
            target: Target percentage (0.0 to 1.0, where 1.0 = 100% of portfolio)
        """
        self._orders.append(
            Order(
                symbol=symbol,
                size=0.0,
                order_type=OrderType.TARGET_PERCENT,
                target_percent=float(target),
            )
        )

    def now(self) -> Optional[pd.Timestamp]:
        """Return the current bar timestamp if available."""
        return self._current_timestamp

