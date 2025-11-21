"""Strategy base class for alphaweave."""

from typing import Any, Dict, List, Optional, Union
from typing import Optional as TypingOptional
from datetime import timedelta

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.core.types import Order, OrderType
from alphaweave.data.events import Event, EventStore
from alphaweave.data.timeframes import resample_frame
from alphaweave.indicators.sma import SMA
from alphaweave.indicators.ema import EMA
from alphaweave.utils.schedule import Schedule


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
        self._resampled_cache: Dict[tuple, pd.Series] = {}  # Cache for resampled series
        self._resample_frame_cache: Dict[tuple[str, str], Frame] = {}  # Cache for resampled frames
        self._schedule: Optional[Any] = None  # Injected by engine
        self._event_store: Optional[Any] = None  # Injected by engine
        # Fast mode support
        self._engine_has_fast_arrays: bool = False
        self._engine_close_np: Optional[Dict[str, Any]] = None

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
            timeframe: Optional timeframe (default: base)

        Returns:
            Close price for the current bar

        Raises:
            ValueError: If _current_index is not set
        """
        # Fast path: use precomputed numpy arrays if available
        if (
            timeframe in (None, "base")
            and self._engine_has_fast_arrays
            and self._engine_close_np is not None
            and symbol is not None
            and symbol in self._engine_close_np
            and isinstance(self._current_index, int)
        ):
            return float(self._engine_close_np[symbol][self._current_index])

        # Fallback: original DataFrame-based logic
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
        
        # Handle timezone compatibility
        if pdf.index.tz is None and ts.tz is not None:
            # Frame is naive, timestamp is aware - convert timestamp to naive
            ts = ts.tz_localize(None)
        elif pdf.index.tz is not None and ts.tz is None:
            # Frame is aware, timestamp is naive - convert timestamp to aware
            ts = ts.tz_localize("UTC")
        elif pdf.index.tz is not None and ts.tz is not None:
            # Both aware - ensure same timezone
            if str(pdf.index.tz) != str(ts.tz):
                ts = ts.tz_convert(pdf.index.tz)
        
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

    def now(self) -> TypingOptional[pd.Timestamp]:
        """
        Return the current bar timestamp if available.
        
        Returns UTC-aware timestamp if available.
        """
        if self._current_timestamp is None:
            return None
        # Ensure UTC-aware
        if self._current_timestamp.tz is None:
            # Assume UTC if naive
            return self._current_timestamp.tz_localize("UTC")
        return self._current_timestamp.tz_convert("UTC") if str(self._current_timestamp.tz) != "UTC" else self._current_timestamp

    def resampled_close(self, symbol: Optional[str] = None, timeframe: str = "1D") -> pd.Series:
        """
        Return a (cached) resampled close series for symbol at timeframe.
        
        Args:
            symbol: Symbol name (required if data is a dict)
            timeframe: Resampling rule (e.g., "1D", "1H", "5T")
            
        Returns:
            Resampled close series as pandas Series
            
        Example:
            daily_close = self.resampled_close("AAPL", "1D")
            daily_sma20 = daily_close.rolling(20).mean()
        """
        # Get base frame
        base_frames = self._timeframes["base"]
        if symbol is None:
            if len(base_frames) == 1:
                symbol = next(iter(base_frames))
            else:
                raise ValueError("symbol required when data contains multiple symbols")
        
        if symbol not in base_frames:
            raise ValueError(f"Symbol '{symbol}' not found in data")
        
        # Cache key
        cache_key = (symbol, timeframe, "close")
        
        if cache_key not in self._resampled_cache:
            # Resample the frame
            base_frame = base_frames[symbol]
            resampled_frame = resample_frame(base_frame, timeframe)
            resampled_df = resampled_frame.to_pandas()
            self._resampled_cache[cache_key] = resampled_df["close"]
        
        return self._resampled_cache[cache_key]

    def is_session_close(self, timeframe: str = "1D") -> bool:
        """
        Check if current bar is the session close for the given timeframe.
        
        Simple implementation: checks if next bar would be in a different period.
        For daily, this means checking if next bar is on a different day.
        
        Args:
            timeframe: Timeframe to check (e.g., "1D", "1H")
            
        Returns:
            True if this is the last bar of the session/period
        """
        if self._current_timestamp is None or self._current_index is None:
            return False
        
        # Get base frame to check next bar
        base_frames = self._timeframes["base"]
        if len(base_frames) == 1:
            symbol = next(iter(base_frames))
        else:
            # For multi-symbol, use first symbol
            symbol = next(iter(base_frames))
        
        base_frame = base_frames[symbol]
        base_df = base_frame.to_pandas()
        
        # Check if there's a next bar
        if self._current_index >= len(base_df) - 1:
            return True  # Last bar
        
        # Get current and next timestamps
        current_ts = base_df.index[self._current_index]
        next_ts = base_df.index[self._current_index + 1]
        
        # Resample both to check if they're in different periods
        # Simple check: for daily, see if dates differ
        if timeframe.endswith("D"):
            return current_ts.date() != next_ts.date()
        elif timeframe.endswith("H"):
            return current_ts.floor("H") != next_ts.floor("H")
        elif timeframe.endswith("T") or timeframe.endswith("min"):
            # For minute bars, check if they're in different periods
            period_minutes = int(timeframe.rstrip("Tmin"))
            current_period = current_ts.floor(f"{period_minutes}T")
            next_period = next_ts.floor(f"{period_minutes}T")
            return current_period != next_period
        
        # Default: assume not session close
        return False

    def clear_indicator_cache(self) -> None:
        """
        Clear the indicator cache.

        Useful for sweeps or when you want to force recomputation.
        """
        self._indicator_cache.clear()

    def get_resampled_frame(self, symbol: str, timeframe: str) -> Frame:
        """
        Get a cached resampled frame for symbol at timeframe.

        Args:
            symbol: Symbol name
            timeframe: Resampling rule (e.g., "1D", "1H", "5T")

        Returns:
            Resampled Frame
        """
        cache_key = (symbol, timeframe)

        if cache_key in self._resample_frame_cache:
            return self._resample_frame_cache[cache_key]

        # Get base frame
        base_frames = self._timeframes["base"]
        if symbol not in base_frames:
            raise ValueError(f"Symbol '{symbol}' not found in data")

        base_frame = base_frames[symbol]
        resampled = resample_frame(base_frame, timeframe)
        self._resample_frame_cache[cache_key] = resampled
        return resampled

    @property
    def schedule(self) -> Schedule:
        """
        Get the Schedule helper for time-based condition checking.

        Returns:
            Schedule instance

        Raises:
            RuntimeError: If schedule has not been initialized by the engine
        """
        if self._schedule is None:
            raise RuntimeError(
                "Schedule not initialized. This should be set by the backtesting engine."
            )
        return self._schedule

    def set_event_store(self, store: Optional[EventStore]) -> None:
        """
        Set the event store for this strategy.

        Called by the engine; not for user override.

        Args:
            store: EventStore instance or None
        """
        self._event_store = store

    def events_now(
        self,
        type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Event]:
        """
        Get events exactly at current Strategy.now().

        Args:
            type: Optional event type filter
            symbol: Optional symbol filter

        Returns:
            List of matching events at current timestamp
        """
        if self._event_store is None:
            return []

        now = self.now()
        if now is None:
            return []

        events = self._event_store.events_at(now)
        return [
            e
            for e in events
            if (type is None or e.type == type)
            and (symbol is None or e.symbol == symbol)
        ]

    def events_window(
        self,
        window: str,
        *,
        type: Optional[str] = None,
        symbol: Optional[str] = None,
        include_current_bar: bool = True,
    ) -> List[Event]:
        """
        Return events in the last `window` period up to now.

        Args:
            window: pandas-style offset string, e.g. "1D", "7D", "2H"
            type: Optional event type filter
            symbol: Optional symbol filter
            include_current_bar: If True, include events at current timestamp

        Returns:
            List of matching events in the window
        """
        if self._event_store is None:
            return []

        now = self.now()
        if now is None:
            return []

        delta = self._parse_window_to_timedelta(window)
        
        # Special case: window="0D" means check only current timestamp
        if delta.total_seconds() == 0:
            events = self._event_store.events_at(now)
            return [
                e
                for e in events
                if (type is None or e.type == type)
                and (symbol is None or e.symbol == symbol)
            ]
        
        start = now - delta
        end = now if include_current_bar else now - timedelta(microseconds=1)

        return list(
            self._event_store.events_in_window(start, end, symbol=symbol, type=type)
        )

    def has_event(
        self,
        *,
        type: str,
        symbol: Optional[str] = None,
        window: str = "0D",
    ) -> bool:
        """
        Convenience method: check if any matching events exist in the window.

        Args:
            type: Event type to check for
            symbol: Optional symbol filter
            window: Time window to check (default: "0D" for current bar only)

        Returns:
            True if any matching events found
        """
        return bool(self.events_window(window, type=type, symbol=symbol))

    def _parse_window_to_timedelta(self, window: str) -> timedelta:
        """
        Parse a window string like "1D", "7D", "2H" into a timedelta.

        Args:
            window: Window string (e.g., "1D", "7D", "2H", "24H")

        Returns:
            timedelta object

        Raises:
            ValueError: If window format is not supported
        """
        window = window.strip().upper()

        if window == "0D" or window == "0H":
            return timedelta(0)

        # Parse patterns like "1D", "7D", "2H", "24H"
        if window.endswith("D"):
            try:
                days = int(window[:-1])
                return timedelta(days=days)
            except ValueError:
                raise ValueError(f"Invalid window format: {window}")

        elif window.endswith("H"):
            try:
                hours = int(window[:-1])
                return timedelta(hours=hours)
            except ValueError:
                raise ValueError(f"Invalid window format: {window}")

        else:
            raise ValueError(
                f"Unsupported window format: {window}. Supported: 'XD' (days) or 'XH' (hours)"
            )

