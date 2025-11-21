# alphaweave — Sprint 0

Weave data. Craft alpha.

## Install (dev)

```bash
python -m pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## Example

```bash
python examples/buy_and_hold.py
```

## Overview

alphaweave is a backtesting framework for Python 3.11+ that provides:

- **Frame abstraction**: Unified interface for Pandas and Polars DataFrames
- **Strategy API**: Simple base class for implementing trading strategies
- **Vector Backtester**: Bar-by-bar backtesting engine with naive execution
- **Data loaders**: CSV and Parquet file loaders with automatic column normalization

## Quick Start

```python
import alphaweave as aw
import pandas as pd

# Create sample data
df = pd.DataFrame({
    "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
    "open": [10+i for i in range(10)],
    "high": [11+i for i in range(10)],
    "low": [9+i for i in range(10)],
    "close": [10+i for i in range(10)],
    "volume": [100]*10,
})

frame = aw.core.frame.Frame.from_pandas(df)

class BuyAndHold(aw.strategy.base.Strategy):
    def init(self): 
        pass
    
    def next(self, i):
        self.order_target_percent("TEST", 1.0)

res = aw.engine.vector.VectorBacktester().run(
    BuyAndHold, 
    data={"TEST": frame}, 
    capital=1000
)

print(res.equity_series)
```

