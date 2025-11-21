"""Tests for Schedule helper."""

from datetime import datetime

import pandas as pd

from alphaweave.utils.schedule import Schedule


def make_now_func(timestamps):
    """Create a now_func that returns timestamps in sequence."""
    index = [0]

    def now_func():
        if index[0] >= len(timestamps):
            return None
        ts = timestamps[index[0]]
        # Don't increment here - Schedule will call multiple times
        return ts

    def increment():
        index[0] += 1

    return now_func, increment


def test_schedule_every_1d():
    """Test every("1D") true on first bar of each day."""
    # Create timestamps: 3 days, 2 bars per day
    timestamps = [
        datetime(2023, 1, 15, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 15, 16, 0, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 16, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 16, 16, 0, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 17, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
    ]

    now_func, increment = make_now_func(timestamps)
    schedule = Schedule(now_func=now_func, index_accessor=lambda: 0)

    results = []
    for _ in timestamps:
        results.append(schedule.every("1D"))
        increment()

    # Should be True on first bar of each day
    assert results == [True, False, True, False, True]


def test_schedule_every_1w():
    """Test every("1W") true on first bar of week."""
    # Create timestamps: 2 weeks, multiple days per week
    timestamps = [
        datetime(2023, 1, 16, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),  # Monday
        datetime(2023, 1, 17, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),  # Tuesday
        datetime(2023, 1, 23, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),  # Next Monday
        datetime(2023, 1, 24, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),  # Next Tuesday
    ]

    now_func, increment = make_now_func(timestamps)
    schedule = Schedule(now_func=now_func, index_accessor=lambda: 0)

    results = []
    for _ in timestamps:
        results.append(schedule.every("1W"))
        increment()

    # Should be True on first bar of each week (Monday)
    assert results == [True, False, True, False]


def test_schedule_every_1m():
    """Test every("1M") true on first bar of month."""
    # Create timestamps: 2 months
    timestamps = [
        datetime(2023, 1, 15, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 20, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 2, 1, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 2, 15, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
    ]

    now_func, increment = make_now_func(timestamps)
    schedule = Schedule(now_func=now_func, index_accessor=lambda: 0)

    results = []
    for _ in timestamps:
        results.append(schedule.every("1M"))
        increment()

    # Should be True on first bar of each month
    assert results == [True, False, True, False]


def test_schedule_at_open():
    """Test at_open() behaves correctly."""
    # Create timestamps: 2 days, multiple bars per day
    timestamps = [
        datetime(2023, 1, 15, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 15, 10, 0, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 16, 9, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 16, 10, 0, 0).replace(tzinfo=pd.Timestamp.now().tz),
    ]

    now_func, increment = make_now_func(timestamps)
    schedule = Schedule(now_func=now_func, index_accessor=lambda: 0)

    results = []
    for _ in timestamps:
        results.append(schedule.at_open())
        increment()

    # Should be True on first bar of each day
    assert results == [True, False, True, False]


def test_schedule_at_time():
    """Test at_time("13:30") works with UTC timestamps."""
    timestamps = [
        datetime(2023, 1, 15, 13, 29, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 15, 13, 30, 0).replace(tzinfo=pd.Timestamp.now().tz),
        datetime(2023, 1, 15, 13, 31, 0).replace(tzinfo=pd.Timestamp.now().tz),
    ]

    now_func, increment = make_now_func(timestamps)
    schedule = Schedule(now_func=now_func, index_accessor=lambda: 0)

    results = []
    for _ in timestamps:
        results.append(schedule.at_time("13:30"))
        increment()

    # Should be True only at 13:30
    assert results == [False, True, False]

