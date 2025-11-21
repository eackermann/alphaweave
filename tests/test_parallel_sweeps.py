"""Tests for parallel parameter sweeps."""

import pandas as pd

from alphaweave.analysis import parameter_sweep
from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.strategy.base import Strategy


class ParamTestStrategy(Strategy):
    """Strategy with configurable parameter."""

    def __init__(self, data, multiplier: float = 1.0):
        super().__init__(data)
        self.multiplier = multiplier

    def init(self):
        pass

    def next(self, i):
        # Simple strategy: buy more if multiplier is higher
        self.order_target_percent("TEST", 0.5 * self.multiplier)


def test_parameter_sweep_sequential():
    """Test parameter sweep with n_jobs=1 (sequential)."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [1000] * 20,
    })
    frame = Frame.from_pandas(df)

    backtester = VectorBacktester()
    param_grid = {
        "multiplier": [1.0, 2.0],
    }

    result = parameter_sweep(
        backtester,
        ParamTestStrategy,
        data={"TEST": frame},
        param_grid=param_grid,
        capital=10000.0,
        n_jobs=1,
    )

    assert len(result.entries) == 2
    assert all(entry.params["multiplier"] in [1.0, 2.0] for entry in result.entries)


def test_parameter_sweep_parallel():
    """Test parameter sweep with n_jobs>1 (parallel)."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [1000] * 20,
    })
    frame = Frame.from_pandas(df)

    backtester = VectorBacktester()
    param_grid = {
        "multiplier": [1.0, 2.0, 3.0],
    }

    # Sequential
    result_seq = parameter_sweep(
        backtester,
        ParamTestStrategy,
        data={"TEST": frame},
        param_grid=param_grid,
        capital=10000.0,
        n_jobs=1,
    )

    # Parallel
    result_par = parameter_sweep(
        backtester,
        ParamTestStrategy,
        data={"TEST": frame},
        param_grid=param_grid,
        capital=10000.0,
        n_jobs=2,
    )

    # Results should be identical (just order might differ)
    assert len(result_seq.entries) == len(result_par.entries) == 3

    # Sort by multiplier to compare
    seq_sorted = sorted(result_seq.entries, key=lambda e: e.params["multiplier"])
    par_sorted = sorted(result_par.entries, key=lambda e: e.params["multiplier"])

    for seq_entry, par_entry in zip(seq_sorted, par_sorted):
        assert seq_entry.params == par_entry.params
        # Results might have slight floating point differences, but should be very close
        assert abs(seq_entry.result.final_equity - par_entry.result.final_equity) < 0.01

