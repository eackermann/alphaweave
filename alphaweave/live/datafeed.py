"""Datafeed abstractions used by LiveRunner."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Protocol, Sequence, Tuple

import pandas as pd

from alphaweave.core.frame import Frame


class DataFeed(Protocol):
    """Minimal streaming interface for live/replay data."""

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def stream(self) -> Iterable[Tuple[pd.Timestamp, Mapping[str, pd.Series]]]:
        """Yield timestamped bar dictionaries."""


class ReplayDataFeed:
    """DataFeed backed by static Frames (useful for replays/tests)."""

    def __init__(self, frames: Mapping[str, Frame]):
        if not frames:
            raise ValueError("ReplayDataFeed requires at least one frame")
        self._frames = frames
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def stream(self) -> Iterable[Tuple[pd.Timestamp, Mapping[str, pd.Series]]]:
        if not self._connected:
            raise RuntimeError("ReplayDataFeed must be connected before streaming")

        pandas_frames = {symbol: frame.to_pandas() for symbol, frame in self._frames.items()}
        calendar = _build_calendar(list(pandas_frames.values()))

        for ts in calendar:
            snapshot: Dict[str, pd.Series] = {}
            for symbol, df in pandas_frames.items():
                snapshot[symbol] = df.loc[ts]
            yield ts, snapshot

    def frames(self) -> Mapping[str, Frame]:
        """Return the original frames (useful for batch engines)."""
        return self._frames


def _build_calendar(frames: Sequence[pd.DataFrame]) -> pd.DatetimeIndex:
    calendar = None
    for df in frames:
        idx = pd.DatetimeIndex(df.index)
        calendar = idx if calendar is None else calendar.intersection(idx)
    if calendar is None or calendar.empty:
        raise ValueError("No overlapping timestamps across frames")
    return calendar.sort_values().unique()


