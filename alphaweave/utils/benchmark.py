"""Benchmark utilities for performance testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import time


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    seconds: float
    result: Any | None = None

    def __repr__(self) -> str:
        return f"BenchmarkResult(name='{self.name}', seconds={self.seconds:.4f})"


def timeit(name: str, func: Callable[[], Any]) -> BenchmarkResult:
    """
    Time a function execution.

    Args:
        name: Name of the benchmark
        func: Function to execute (no arguments)

    Returns:
        BenchmarkResult with timing and function result
    """
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return BenchmarkResult(name=name, seconds=end - start, result=result)

