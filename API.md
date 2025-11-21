# alphaweave API Documentation

**Version:** 0.0.1  
**Python:** 3.11+

Complete API reference for the alphaweave backtesting framework.

---

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Frame Abstraction](#frame-abstraction)
- [Data Loaders](#data-loaders)
- [Strategy API](#strategy-api)
- [Backtesting Engine](#backtesting-engine)
- [Results](#results)
- [Utilities](#utilities)

---

## Overview

alphaweave provides a minimal but production-ready backtesting framework with:

- **Unified data interface**: Works with both Pandas and Polars DataFrames
- **Simple strategy API**: Minimal base class for implementing trading strategies
- **Vector backtester**: Bar-by-bar execution engine with naive order execution
- **Type-safe core**: Dataclasses for all core trading concepts

---

## Core Types

### `alphaweave.core.types.Bar`

OHLCV bar data container.

```python
@dataclass
class Bar:
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    symbol: Optional[str] = None
```

**Attributes:**
- `datetime` (datetime): Bar timestamp
- `open` (float): Opening price
- `high` (float): High price
- `low` (float): Low price
- `close` (float): Closing price
- `volume` (Optional[float]): Trading volume
- `symbol` (Optional[str]): Symbol identifier

**Example:**
```python
from alphaweave.core.types import Bar
from datetime import datetime

bar = Bar(
    datetime=datetime(2020, 1, 1),
    open=100.0,
    high=105.0,
    low=99.0,
    close=103.0,
    volume=1000000.0,
    symbol="AAPL"
)
```

---

### `alphaweave.core.types.Order`

Order request container.

```python
@dataclass
class Order:
    id: int
    symbol: str
    size: float  # positive = buy, negative = sell
    price: Optional[float] = None
    order_type: str = "market"
```

**Attributes:**
- `id` (int): Unique order identifier
- `symbol` (str): Symbol to trade
- `size` (float): Order size (positive = buy, negative = sell)
- `price` (Optional[float]): Limit price (if applicable)
- `order_type` (str): Order type, default "market"

---

### `alphaweave.core.types.Fill`

Executed order fill record.

```python
@dataclass
class Fill:
    order_id: int
    symbol: str
    size: float
    price: float
    datetime: datetime
```

**Attributes:**
- `order_id` (int): Associated order ID
- `symbol` (str): Traded symbol
- `size` (float): Fill size
- `price` (float): Execution price
- `datetime` (datetime): Fill timestamp

---

### `alphaweave.core.types.Position`

Portfolio position container.

```python
@dataclass
class Position:
    symbol: str
    size: float
    avg_price: float
```

**Attributes:**
- `symbol` (str): Position symbol
- `size` (float): Position size (positive = long, negative = short)
- `avg_price` (float): Average entry price

---

## Frame Abstraction

### `alphaweave.core.frame.Frame`

Canonical Frame abstraction wrapping a pandas or polars DataFrame. Provides a unified interface for working with time-series OHLCV data regardless of the underlying DataFrame library.

#### Constructor

```python
Frame(backend_obj: Any)
```

Initialize Frame with a pandas or polars DataFrame.

**Parameters:**
- `backend_obj`: pandas DataFrame or polars DataFrame

**Raises:**
- `TypeError`: If backend_obj is not a pandas or polars DataFrame

---

#### Class Methods

##### `Frame.from_pandas(df: pd.DataFrame) -> Frame`

Create Frame from pandas DataFrame and validate/normalize columns.

**Parameters:**
- `df` (pd.DataFrame): pandas DataFrame with OHLCV data

**Returns:**
- `Frame`: Frame instance with normalized columns and datetime index

**Raises:**
- `ValueError`: If required columns are missing or datetime cannot be determined

**Example:**
```python
import pandas as pd
from alphaweave.core.frame import Frame

df = pd.DataFrame({
    "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
    "open": [100 + i for i in range(10)],
    "high": [101 + i for i in range(10)],
    "low": [99 + i for i in range(10)],
    "close": [100 + i for i in range(10)],
    "volume": [1000000] * 10
})

frame = Frame.from_pandas(df)
```

---

##### `Frame.from_polars(pl_df: pl.DataFrame) -> Frame`

Create Frame from polars DataFrame and validate/normalize columns.

**Parameters:**
- `pl_df` (pl.DataFrame): polars DataFrame with OHLCV data

**Returns:**
- `Frame`: Frame instance with normalized columns and datetime index

**Raises:**
- `ValueError`: If required columns are missing or datetime cannot be determined

**Example:**
```python
import polars as pl
from alphaweave.core.frame import Frame

pl_df = pl.DataFrame({
    "datetime": pl.date_range("2020-01-01", "2020-01-10", "1d"),
    "open": [100 + i for i in range(10)],
    "close": [100 + i for i in range(10)],
    # ... other columns
})

frame = Frame.from_polars(pl_df)
```

---

#### Instance Methods

##### `to_pandas() -> pd.DataFrame`

Convert Frame to pandas DataFrame.

**Returns:**
- `pd.DataFrame`: pandas DataFrame with datetime index

**Note:** The returned DataFrame will have a datetime index. If the Frame was created from polars, the datetime column will be converted to the index.

---

##### `to_polars() -> pl.DataFrame`

Convert Frame to polars DataFrame.

**Returns:**
- `pl.DataFrame`: polars DataFrame with datetime as a column

**Note:** The returned DataFrame will have datetime as a column (polars doesn't support named indexes). The datetime column is named "datetime".

---

##### `validate() -> None`

Raise ValueError if datetime column/index missing or no required columns.

**Raises:**
- `ValueError`: If validation fails (missing datetime index or required OHLC columns)

**Required columns:**
- `open`, `high`, `low`, `close` (minimum)
- `volume` (optional but recommended)

**Column normalization:**
The Frame automatically normalizes column names:
- Datetime variations: `timestamp`, `dt`, `date`, `time` → `datetime`
- OHLCV variations: `o` → `open`, `h` → `high`, `l` → `low`, `c` → `close`, `v`/`vol` → `volume`
- All column names are lowercased

---

## Data Loaders

### `alphaweave.data.loaders.load_csv`

```python
load_csv(path: str, symbol: Optional[str] = None) -> Frame
```

Load CSV file into a Frame. Automatically normalizes column names to canonical form.

**Parameters:**
- `path` (str): Path to CSV file
- `symbol` (Optional[str]): Optional symbol name to add as a column to the frame

**Returns:**
- `Frame`: Frame instance with normalized columns and datetime index

**Example:**
```python
from alphaweave.data.loaders import load_csv

# Load CSV with automatic column normalization
frame = load_csv("data/AAPL.csv", symbol="AAPL")

# Access as pandas DataFrame
df = frame.to_pandas()
```

**Supported column name variations:**
- Datetime: `Timestamp`, `timestamp`, `dt`, `date`, `time`
- OHLCV: `Open`/`open`, `High`/`high`, `Low`/`low`, `Close`/`close`, `Volume`/`volume`

---

### `alphaweave.data.loaders.load_parquet`

```python
load_parquet(path: str, symbol: Optional[str] = None) -> Frame
```

Load Parquet file into a Frame. Automatically normalizes column names.

**Parameters:**
- `path` (str): Path to Parquet file
- `symbol` (Optional[str]): Optional symbol name to add as a column to the frame

**Returns:**
- `Frame`: Frame instance with normalized columns and datetime index

**Example:**
```python
from alphaweave.data.loaders import load_parquet

frame = load_parquet("data/AAPL.parquet", symbol="AAPL")
```

---

## Strategy API

### `alphaweave.strategy.base.Strategy`

Base class for implementing trading strategies. Subclasses must implement the `next()` method.

#### Constructor

```python
Strategy(data: Union[Frame, Dict[str, Frame]])
```

Initialize strategy with data.

**Parameters:**
- `data`: Either a single Frame or a dictionary mapping symbol names to Frame objects

**Example:**
```python
from alphaweave.strategy.base import Strategy
from alphaweave.core.frame import Frame

# Single asset
strategy = MyStrategy(frame)

# Multiple assets
strategy = MyStrategy({"AAPL": frame_aapl, "MSFT": frame_msft})
```

---

#### Methods

##### `init() -> None`

Called once before the backtest loop starts. Override this method to perform initialization tasks.

**Example:**
```python
class MyStrategy(Strategy):
    def init(self):
        self.sma_period = 20
        self.position = 0.0
```

---

##### `next(index: Any) -> None`

Called once per bar during the backtest loop. **Must be implemented by subclasses.**

**Parameters:**
- `index`: Current bar index (integer or pandas.Timestamp)

**Example:**
```python
class MyStrategy(Strategy):
    def next(self, i):
        # Access data
        df = self.data["AAPL"].to_pandas()
        current_price = df.iloc[i]["close"]
        
        # Place orders
        if current_price > 100:
            self.order_target_percent("AAPL", 1.0)  # Go 100% long
        else:
            self.order_target_percent("AAPL", 0.0)  # Exit position
```

---

##### `order_target_percent(symbol: str, target: float) -> None`

Request to set target percentage of portfolio in a symbol.

**Parameters:**
- `symbol` (str): Symbol to target
- `target` (float): Target percentage (0.0 to 1.0, where 1.0 = 100% of portfolio)

**Example:**
```python
# Allocate 50% of portfolio to AAPL
self.order_target_percent("AAPL", 0.5)

# Exit position (0%)
self.order_target_percent("AAPL", 0.0)

# Go 100% long
self.order_target_percent("AAPL", 1.0)
```

**Note:** Orders are collected during each bar and executed at the end of the bar at the close price. The backtester will automatically calculate the required position size based on current portfolio equity.

---

## Backtesting Engine

### `alphaweave.engine.base.BaseBacktester`

Protocol defining the interface for backtesters.

```python
class BaseBacktester(Protocol):
    def run(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, Frame],
        capital: float = 100_000.0,
    ) -> BacktestResult:
        ...
```

**Parameters:**
- `strategy_cls`: Strategy class to instantiate
- `data`: Dictionary mapping symbol names to Frame objects
- `capital`: Starting capital (default: 100,000)

**Returns:**
- `BacktestResult`: Backtest results with equity series and trades

---

### `alphaweave.engine.vector.VectorBacktester`

Vector backtester with naive bar-by-bar execution. Implements the `BaseBacktester` protocol.

#### Execution Model

The VectorBacktester implements a simple execution model:

1. **Strategy instantiation**: Creates an instance of the strategy class with the provided data
2. **Initialization**: Calls `strategy.init()` once before the loop
3. **Bar-by-bar loop**:
   - Clears strategy orders for the current bar
   - Calls `strategy.next(index)` to allow strategy to place orders
   - Processes orders:
     - For `order_target_percent`: Calculates target position size based on current portfolio equity and symbol price
     - Executes immediately at bar close price (no slippage, no fees)
   - Updates portfolio (cash, positions)
   - Records equity value for the bar
4. **Returns**: `BacktestResult` with equity series and trades

#### Simplifications (Sprint 0)

- **No slippage**: Orders execute exactly at the close price
- **No fees**: No commission or transaction costs
- **No partial fills**: Orders execute fully or not at all
- **No margin/leverage**: Cash-only trading
- **Single asset support**: Multi-asset support is basic (all frames must have the same index)

---

#### Methods

##### `run(strategy_cls, data, capital=100_000.0) -> BacktestResult`

Run the backtest and return results.

**Parameters:**
- `strategy_cls` (Type[Strategy]): Strategy class to instantiate
- `data` (Dict[str, Frame]): Dictionary mapping symbol names to Frame objects
- `capital` (float): Starting capital (default: 100,000.0)

**Returns:**
- `BacktestResult`: Results containing equity series and trades

**Example:**
```python
from alphaweave.engine.vector import VectorBacktester
from alphaweave.core.frame import Frame

# Prepare data
data = {
    "AAPL": frame_aapl,
    "MSFT": frame_msft
}

# Run backtest
backtester = VectorBacktester()
result = backtester.run(MyStrategy, data, capital=100000.0)

# Access results
print(f"Final equity: ${result.equity_series[-1]:,.2f}")
print(f"Number of trades: {len(result.trades)}")
```

**Portfolio Management:**
- The backtester maintains cash and positions
- Positions are tracked with average entry price
- Equity is calculated as: `cash + sum(position_size * current_price)`
- Orders are limited by available cash (for buys) or position size (for sells)

---

## Results

### `alphaweave.results.result.BacktestResult`

Container for backtest results.

#### Constructor

```python
BacktestResult(equity_series: List[float], trades: List[Any])
```

**Parameters:**
- `equity_series` (List[float]): List of equity values, one per bar
- `trades` (List[Any]): List of trade/fill records (Fill objects)

---

#### Attributes

- `equity_series` (List[float]): Equity curve values (one per bar)
- `trades` (List[Fill]): List of executed trades (Fill objects)

---

#### Methods

##### `plot_equity() -> None`

Plot the equity curve using matplotlib.

**Example:**
```python
result = backtester.run(MyStrategy, data)
result.plot_equity()  # Opens matplotlib window
```

---

## Utilities

### `alphaweave.utils.time.ensure_datetime_index`

```python
ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame
```

Convert datetime column to pandas datetime and set as index if not already.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame with datetime index

**Raises:**
- `ValueError`: If no datetime column or index can be found

**Behavior:**
1. If index is already a DatetimeIndex, returns DataFrame unchanged
2. Looks for datetime columns: `datetime`, `dt`, `timestamp`
3. Converts found column to datetime and sets as index
4. If no column found, attempts to convert existing index to datetime

**Example:**
```python
from alphaweave.utils.time import ensure_datetime_index
import pandas as pd

df = pd.DataFrame({
    "datetime": ["2020-01-01", "2020-01-02"],
    "close": [100, 101]
})

df = ensure_datetime_index(df)
# df now has DatetimeIndex
```

---

## Usage Examples

### Complete Example: Buy-and-Hold Strategy

```python
import alphaweave as aw
import pandas as pd

# Create sample data
df = pd.DataFrame({
    "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
    "open": [10 + i for i in range(10)],
    "high": [11 + i for i in range(10)],
    "low": [9 + i for i in range(10)],
    "close": [10 + i for i in range(10)],
    "volume": [100] * 10,
})

# Create Frame
frame = aw.core.frame.Frame.from_pandas(df)

# Define strategy
class BuyAndHold(aw.strategy.base.Strategy):
    def init(self):
        pass
    
    def next(self, i):
        # Buy and hold on first bar
        self.order_target_percent("TEST", 1.0)

# Run backtest
backtester = aw.engine.vector.VectorBacktester()
result = backtester.run(
    BuyAndHold,
    data={"TEST": frame},
    capital=1000.0
)

# Print results
print(f"Equity series: {result.equity_series}")
print(f"Number of trades: {len(result.trades)}")
print(f"Final equity: ${result.equity_series[-1]:,.2f}")
```

### Loading Data from CSV

```python
from alphaweave.data.loaders import load_csv

# Load CSV with automatic normalization
frame = load_csv("data/AAPL.csv", symbol="AAPL")

# Convert to pandas for analysis
df = frame.to_pandas()
print(df.head())
```

### Multi-Asset Strategy

```python
from alphaweave.strategy.base import Strategy
from alphaweave.core.frame import Frame

class DualAssetStrategy(Strategy):
    def init(self):
        self.aapl_weight = 0.6
        self.msft_weight = 0.4
    
    def next(self, i):
        # Allocate 60% to AAPL, 40% to MSFT
        self.order_target_percent("AAPL", self.aapl_weight)
        self.order_target_percent("MSFT", self.msft_weight)

# Prepare data
data = {
    "AAPL": frame_aapl,
    "MSFT": frame_msft
}

# Run backtest
result = VectorBacktester().run(DualAssetStrategy, data)
```

---

## Type Hints

All public APIs include type hints for better IDE support and type checking:

```python
from typing import Optional, Dict, List, Type, Union
from datetime import datetime
import pandas as pd
import polars as pl
```

---

## Error Handling

### Common Exceptions

**`ValueError`**: Raised when:
- Frame validation fails (missing required columns or datetime)
- Data format is invalid

**`TypeError`**: Raised when:
- Wrong type passed to Frame constructor
- Invalid argument types

**`NotImplementedError`**: Raised when:
- Strategy subclass doesn't implement `next()`

---

## Notes

- **Datetime handling**: All Frames must have a datetime index. The Frame abstraction automatically handles conversion between pandas (index) and polars (column) representations.
- **Column normalization**: Column names are automatically normalized to lowercase canonical forms (e.g., "Open" → "open", "Timestamp" → "datetime").
- **Order execution**: Orders execute at the close price of the current bar with no slippage or fees (Sprint 0 simplification).
- **Portfolio tracking**: Positions are tracked with average entry price using weighted averages when adding to existing positions.

---

## Version

This documentation corresponds to **alphaweave version 0.0.1** (Sprint 0).

