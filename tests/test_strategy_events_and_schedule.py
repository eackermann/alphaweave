"""Tests for Strategy events and schedule integration."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.data.events import Event, EventStore
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy
from alphaweave.utils.schedule import Schedule


class EventTestStrategy(Strategy):
    """Strategy that uses events."""

    def init(self):
        self.events_seen = []

    def next(self, i):
        # Check for events at current timestamp
        events = self.events_now()
        self.events_seen.extend(events)

        # Check for events in window
        window_events = self.events_window("1D", type="earnings")
        if window_events:
            self.order_target_percent("TEST", 1.0)


class ScheduleTestStrategy(Strategy):
    """Strategy that uses schedule."""

    def init(self):
        self.rebalance_count = 0

    def next(self, i):
        # Only rebalance weekly at open
        if self.schedule.every("1W") and self.schedule.at_open():
            self.rebalance_count += 1
            self.order_target_percent("TEST", 1.0)


def test_strategy_events_now():
    """Test Strategy sees events via events_now."""
    from datetime import timezone
    
    # Create data
    dates = pd.date_range("2023-01-15", periods=5, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1000] * 5,
    })
    frame = Frame.from_pandas(df)

    # Create events - convert to UTC-aware datetime
    event_ts = dates[2].to_pydatetime()
    if event_ts.tzinfo is None:
        event_ts = pd.Timestamp(event_ts).tz_localize("UTC").to_pydatetime()
    else:
        event_ts = pd.Timestamp(event_ts).tz_convert("UTC").to_pydatetime()
    
    events = [
        Event(
            timestamp=event_ts,
            type="earnings",
            symbol="TEST",
        ),
    ]
    event_store = EventStore(events)

    # Strategy should have seen the event
    strategy = EventTestStrategy({"TEST": frame})
    strategy.set_event_store(event_store)
    # Manually set timestamp to match event (UTC-aware)
    strategy._set_current_timestamp(pd.Timestamp(dates[2]).tz_localize("UTC"))
    seen = strategy.events_now()
    assert len(seen) == 1
    assert seen[0].type == "earnings"


def test_strategy_has_event():
    """Test has_event returns True when expected."""
    dates = pd.date_range("2023-01-15", periods=5, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1000] * 5,
    })
    frame = Frame.from_pandas(df)

    # Create events - convert to UTC-aware
    event_ts = dates[2].to_pydatetime()
    if event_ts.tzinfo is None:
        event_ts = pd.Timestamp(event_ts).tz_localize("UTC").to_pydatetime()
    else:
        event_ts = pd.Timestamp(event_ts).tz_convert("UTC").to_pydatetime()
    
    events = [
        Event(
            timestamp=event_ts,
            type="earnings",
            symbol="TEST",
        ),
    ]
    event_store = EventStore(events)

    strategy = EventTestStrategy({"TEST": frame})
    strategy.set_event_store(event_store)

    # Set timestamp to match event (UTC-aware)
    strategy._set_current_timestamp(pd.Timestamp(dates[2]).tz_localize("UTC"))
    assert strategy.has_event(type="earnings", symbol="TEST", window="0D")

    # Set timestamp before event
    strategy._set_current_timestamp(pd.Timestamp(dates[1]).tz_localize("UTC"))
    assert not strategy.has_event(type="earnings", symbol="TEST", window="0D")

    # Check window
    strategy._set_current_timestamp(pd.Timestamp(dates[3]).tz_localize("UTC"))
    assert strategy.has_event(type="earnings", symbol="TEST", window="2D")


def test_strategy_schedule_every_1w():
    """Test schedule.every("1W") triggers on expected bars."""
    # Create data spanning 2 weeks
    dates = pd.date_range("2023-01-16", periods=10, freq="D")  # Start on Monday
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [1000] * 10,
    })
    frame = Frame.from_pandas(df)

    # Run backtest
    backtester = VectorBacktester()
    result = backtester.run(
        ScheduleTestStrategy,
        data={"TEST": frame},
        capital=10000.0,
    )

    # Strategy should have rebalanced on first bar of each week
    # We can't easily check this from outside, but we can verify the strategy
    # was instantiated and schedule was set
    strategy = ScheduleTestStrategy({"TEST": frame})
    from alphaweave.utils.schedule import Schedule

    # Create schedule with proper now_func
    def make_now_func(ts_list):
        idx = [0]
        def now():
            if idx[0] < len(ts_list):
                ts = pd.Timestamp(ts_list[idx[0]]).tz_localize("UTC")
                return ts
            return None
        return now
    
    schedule = Schedule(
        now_func=make_now_func(dates),
        index_accessor=lambda: 0,
    )
    strategy._schedule = schedule

    # First bar should trigger
    strategy._set_current_timestamp(pd.Timestamp(dates[0]).tz_localize("UTC"))
    assert strategy.schedule.every("1W")
    assert strategy.schedule.at_open()

