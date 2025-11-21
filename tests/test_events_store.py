"""Tests for EventStore and event loading."""

from datetime import datetime

import pandas as pd

from alphaweave.data.events import Event, EventStore, load_events_csv


def test_event_creation():
    """Test Event dataclass creation."""
    from datetime import timezone
    
    event = Event(
        timestamp=datetime(2023, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
        type="earnings",
        symbol="AAPL",
        payload={"eps": 1.5, "revenue": 1000000},
    )

    assert event.type == "earnings"
    assert event.symbol == "AAPL"
    assert event.payload["eps"] == 1.5


def test_event_requires_timezone():
    """Test that Event requires timezone-aware timestamp."""
    try:
        Event(
            timestamp=datetime(2023, 1, 15, 16, 0, 0),  # Naive
            type="earnings",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_eventstore_events_at():
    """Test EventStore.events_at() returns exact matches."""
    from datetime import timezone
    
    events = [
        Event(
            timestamp=datetime(2023, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="AAPL",
        ),
        Event(
            timestamp=datetime(2023, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
            type="news",
            symbol="AAPL",
        ),
        Event(
            timestamp=datetime(2023, 1, 16, 10, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="MSFT",
        ),
    ]

    store = EventStore(events)

    # Query exact timestamp
    from datetime import timezone
    query_ts = datetime(2023, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
    found = store.events_at(query_ts)

    assert len(found) == 2
    assert all(e.type in ("earnings", "news") for e in found)


def test_eventstore_events_in_window():
    """Test EventStore.events_in_window() respects [start, end) and filters."""
    from datetime import timezone
    
    events = [
        Event(
            timestamp=datetime(2023, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="AAPL",
        ),
        Event(
            timestamp=datetime(2023, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
            type="news",
            symbol="AAPL",
        ),
        Event(
            timestamp=datetime(2023, 1, 16, 10, 0, 0, tzinfo=timezone.utc),
            type="earnings",
            symbol="MSFT",
        ),
        Event(
            timestamp=datetime(2023, 1, 17, 10, 0, 0, tzinfo=timezone.utc),
            type="macro",
            symbol=None,
        ),
    ]

    store = EventStore(events)

    # Query window [2023-01-15 00:00, 2023-01-17 00:00)
    start = datetime(2023, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2023, 1, 17, 0, 0, 0, tzinfo=timezone.utc)

    # All events in window
    all_events = store.events_in_window(start, end)
    assert len(all_events) == 3  # Excludes the 2023-01-17 event

    # Filter by type
    earnings = store.events_in_window(start, end, type="earnings")
    assert len(earnings) == 2

    # Filter by symbol
    aapl_events = store.events_in_window(start, end, symbol="AAPL")
    assert len(aapl_events) == 2

    # Filter by both
    aapl_earnings = store.events_in_window(start, end, type="earnings", symbol="AAPL")
    assert len(aapl_earnings) == 1


def test_load_events_csv():
    """Test load_events_csv correctly parses timestamps, payload, and tz."""
    import tempfile
    import os

    # Create temporary CSV
    csv_content = """timestamp,type,symbol,eps,revenue
2023-01-15 16:00:00,earnings,AAPL,1.5,1000000
2023-01-16 10:00:00,earnings,MSFT,2.0,2000000
2023-01-17 12:00:00,macro,,
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        store = load_events_csv(temp_path)

        events = list(store._events)
        assert len(events) == 3

        # Check first event
        aapl_event = [e for e in events if e.symbol == "AAPL"][0]
        assert aapl_event.type == "earnings"
        assert aapl_event.payload is not None
        assert aapl_event.payload["eps"] == 1.5
        assert aapl_event.payload["revenue"] == 1000000

        # Check macro event (no symbol)
        macro_event = [e for e in events if e.symbol is None][0]
        assert macro_event.type == "macro"

    finally:
        os.unlink(temp_path)

