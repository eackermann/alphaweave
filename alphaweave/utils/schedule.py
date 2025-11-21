"""Scheduling helpers for strategy execution timing."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd


class Schedule:
    """
    Scheduling helper for determining when certain time-based conditions are met.

    Used by strategies to check if they should execute logic on specific
    time boundaries (daily, weekly, monthly) or at specific times (open, close).
    """

    def __init__(
        self,
        now_func: Callable[[], Optional[datetime]],
        index_accessor: Callable[[], any],
    ):
        """
        Initialize Schedule helper.

        Args:
            now_func: Function that returns current timestamp (typically Strategy.now)
            index_accessor: Function that returns the current index/position in the time series
        """
        self._now_func = now_func
        self._index_accessor = index_accessor
        self._prev_date: Optional[datetime.date] = None
        self._prev_week: Optional[tuple[int, int]] = None  # (year, week)
        self._prev_month: Optional[tuple[int, int]] = None  # (year, month)
        self._current_index: Optional[any] = None

    def _get_now(self) -> datetime:
        """Get current timestamp, ensuring UTC-aware."""
        now = self._now_func()
        if now is None:
            raise RuntimeError("Current timestamp not available")
        if now.tzinfo is None:
            # Assume UTC if naive
            now = now.replace(tzinfo=pd.Timestamp.now().tz)
        return now

    def every(self, freq: str) -> bool:
        """
        True only on bars that mark the start of a new period.

        Supported frequencies:
            - "1D": First bar of each day
            - "1W": First bar of each week (Monday)
            - "1M": First bar of each month

        Args:
            freq: Frequency string ("1D", "1W", "1M")

        Returns:
            True if this bar marks the start of a new period
        """
        now = self._get_now()
        date = now.date()

        if freq == "1D":
            # First bar of each day
            if self._prev_date is None:
                self._prev_date = date
                return True
            is_new_day = date != self._prev_date
            self._prev_date = date
            return is_new_day

        elif freq == "1W":
            # First bar of each week (ISO week)
            year, week, _ = date.isocalendar()
            week_key = (year, week)
            if self._prev_week is None:
                self._prev_week = week_key
                return True
            is_new_week = week_key != self._prev_week
            self._prev_week = week_key
            return is_new_week

        elif freq == "1M":
            # First bar of each month
            month_key = (date.year, date.month)
            if self._prev_month is None:
                self._prev_month = month_key
                return True
            is_new_month = month_key != self._prev_month
            self._prev_month = month_key
            return is_new_month

        else:
            raise ValueError(f"Unsupported frequency: {freq}")

    def at_open(self) -> bool:
        """
        True on the first bar of the trading session (for the current day).

        In absence of explicit exchange calendars, treats the first bar of a UTC day
        as session open.

        Returns:
            True if this is the first bar of the current day
        """
        now = self._get_now()
        date = now.date()

        # Check if this is the first bar of the day
        # We do this by checking if previous date was different
        if self._prev_date is None:
            self._prev_date = date
            return True

        is_first_bar_of_day = date != self._prev_date
        if is_first_bar_of_day:
            self._prev_date = date
        return is_first_bar_of_day

    def at_close(self, calendar: Optional[any] = None) -> bool:
        """
        True on the last bar of the trading session (for the current day).

        By default: last bar of that UTC day. If calendar is provided, checks
        if the next bar would be on a different day.

        Args:
            calendar: Optional calendar/index to check next bar (if available)

        Returns:
            True if this appears to be the last bar of the current day
        """
        now = self._get_now()
        current_index = self._index_accessor()
        
        # If we have calendar access, check if next bar is on different day
        if calendar is not None:
            try:
                idx = self._index_accessor()
                if isinstance(idx, int) and hasattr(calendar, '__len__'):
                    if idx < len(calendar) - 1:
                        next_ts = calendar[idx + 1]
                        if isinstance(next_ts, pd.Timestamp):
                            next_date = next_ts.date()
                            current_date = now.date()
                            return current_date != next_date
            except (IndexError, AttributeError, TypeError):
                pass
        
        # Fallback heuristic: if time is 23:00 or later, consider it close
        # This is a simple approximation
        return now.hour >= 23 or (now.hour == 22 and now.minute >= 30)

    def at_time(self, hhmm: str) -> bool:
        """
        True when current bar's time matches hh:mm in UTC.

        Args:
            hhmm: Time string in "HH:MM" format (interpreted as UTC)

        Returns:
            True if current bar's time matches the specified time
        """
        now = self._get_now()
        try:
            hour, minute = map(int, hhmm.split(":"))
            return now.hour == hour and now.minute == minute
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid time format: {hhmm}. Expected 'HH:MM'")

