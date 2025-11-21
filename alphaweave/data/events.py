"""Event store and loading for alphaweave."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from alphaweave.core.frame import Frame


@dataclass(frozen=True)
class Event:
    """
    Generic event representation.

    Attributes:
        timestamp: Event timestamp (must be UTC-aware)
        type: Event type (e.g., "earnings", "news", "macro")
        symbol: Symbol associated with event (None for macro events)
        payload: Arbitrary metadata dictionary
    """

    timestamp: datetime  # UTC-aware
    type: str
    symbol: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        """Validate that timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            raise ValueError("Event timestamp must be timezone-aware (preferably UTC)")

    def __repr__(self) -> str:
        sym_str = f", symbol={self.symbol}" if self.symbol else ""
        payload_str = f", payload={self.payload}" if self.payload else ""
        return (
            f"Event(timestamp={self.timestamp}, type={self.type}{sym_str}{payload_str})"
        )


class EventStore:
    """
    Read-only event container with efficient timestamp-based queries.

    Events are indexed by timestamp and optionally by symbol for fast lookups.
    """

    def __init__(self, events: Iterable[Event]):
        """
        Initialize EventStore with a collection of events.

        Args:
            events: Iterable of Event objects
        """
        self._events: list[Event] = list(events)
        self._events.sort(key=lambda e: e.timestamp)

        # Index by timestamp (date-normalized for exact matches)
        self._by_timestamp: Dict[datetime, list[Event]] = defaultdict(list)
        for event in self._events:
            # Normalize to date for exact timestamp matching
            date_key = event.timestamp.date()
            # Store with original timestamp for precise matching
            self._by_timestamp[event.timestamp] = self._by_timestamp.get(
                event.timestamp, []
            ) + [event]

        # Index by symbol for faster filtering
        self._by_symbol: Dict[Optional[str], list[Event]] = defaultdict(list)
        for event in self._events:
            self._by_symbol[event.symbol].append(event)

    def events_at(self, timestamp: datetime) -> Sequence[Event]:
        """
        Get all events that occur exactly at timestamp.

        Args:
            timestamp: Query timestamp (should be UTC-aware)

        Returns:
            List of events at the exact timestamp
        """
        # For exact matching, we need to handle timezone-aware comparisons
        if timestamp.tzinfo is None:
            # If naive, assume UTC
            timestamp = pd.Timestamp(timestamp).tz_localize("UTC").to_pydatetime()
        else:
            # Ensure UTC for comparison
            timestamp = pd.Timestamp(timestamp).tz_convert("UTC").to_pydatetime()

        # Find events with matching timestamp (within microsecond precision)
        matching = []
        for event in self._events:
            # Normalize event timestamp to UTC for comparison
            event_ts = pd.Timestamp(event.timestamp).tz_convert("UTC").to_pydatetime()
            
            if event_ts == timestamp:
                matching.append(event)
            # Also match if dates are the same and within 1 second (for day-level granularity)
            elif (
                event_ts.date() == timestamp.date()
                and abs((event_ts - timestamp).total_seconds()) < 1.0
            ):
                matching.append(event)

        return matching

    def events_in_window(
        self,
        start: datetime,
        end: datetime,
        *,
        symbol: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Sequence[Event]:
        """
        Get events in [start, end), optionally filtered by symbol and type.

        Args:
            start: Start of time window (inclusive)
            end: End of time window (exclusive)
            symbol: Optional symbol filter
            type: Optional event type filter

        Returns:
            List of matching events
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=pd.Timestamp.now().tz)
        if end.tzinfo is None:
            end = end.replace(tzinfo=pd.Timestamp.now().tz)

        matching = []
        for event in self._events:
            # Check time window [start, end)
            if event.timestamp < start or event.timestamp >= end:
                continue

            # Apply filters
            if symbol is not None and event.symbol != symbol:
                continue
            if type is not None and event.type != type:
                continue

            matching.append(event)

        return matching


def load_events_csv(path: str, tz: Optional[str] = None) -> EventStore:
    """
    Load events from a CSV file into an EventStore.

    Expected CSV format:
        timestamp,type,symbol,...
        2023-01-15 16:00:00,earnings,AAPL,...
        2023-02-01 10:00:00,macro,,
        ...

    Columns:
        - timestamp: datetime-like (required)
        - type: event type string (required)
        - symbol: symbol string (optional, empty for macro events)
        - ... any extra columns become payload dict

    Args:
        path: Path to CSV file
        tz: Optional timezone to localize naive timestamps before converting to UTC

    Returns:
        EventStore containing all loaded events
    """
    df = pd.read_csv(path)

    # Normalize column names (case-insensitive)
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("timestamp", "datetime", "date", "time"):
            rename_map[col] = "timestamp"
        elif lc in ("type", "event_type", "kind"):
            rename_map[col] = "type"
        elif lc in ("symbol", "sym", "ticker"):
            rename_map[col] = "symbol"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Validate required columns
    if "timestamp" not in df.columns:
        raise ValueError("CSV missing required 'timestamp' column")
    if "type" not in df.columns:
        raise ValueError("CSV missing required 'type' column")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Handle timezone: localize to tz if provided, then convert to UTC
    if tz is not None:
        # Check if timestamps are naive
        if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
            # Already timezone-aware
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        else:
            # Naive, localize then convert
            df["timestamp"] = df["timestamp"].dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        # Check if timestamps are naive
        if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
            # Already timezone-aware, convert to UTC if needed
            if str(df["timestamp"].dtype.tz) != "UTC":
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        else:
            # Naive, assume UTC
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Extract payload columns (everything except timestamp, type, symbol)
    payload_cols = [
        col
        for col in df.columns
        if col not in ("timestamp", "type", "symbol")
    ]

    events = []
    for _, row in df.iterrows():
        # Build payload from extra columns
        payload = None
        if payload_cols:
            payload = {
                col: row[col] for col in payload_cols if pd.notna(row[col])
            }
            if not payload:
                payload = None

        event = Event(
            timestamp=row["timestamp"].to_pydatetime(),
            type=str(row["type"]),
            symbol=str(row["symbol"]).upper() if pd.notna(row.get("symbol")) else None,
            payload=payload,
        )
        events.append(event)

    return EventStore(events)

