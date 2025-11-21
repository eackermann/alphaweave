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
- [Corporate Actions](#corporate-actions)
- [Strategy API](#strategy-api)
- [Factor Library & Pipeline API](#factor-library--pipeline-api)
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

## Corporate Actions

### `alphaweave.data.corporate_actions.SplitAction`

Represents a stock split action.

```python
@dataclass
class SplitAction:
    symbol: str
    date: datetime
    ratio: float  # e.g., 2.0 for a 2-for-1 split
```

**Attributes:**
- `symbol` (str): Stock symbol
- `date` (datetime): Split date
- `ratio` (float): Split ratio (e.g., 2.0 for 2-for-1, 4.0 for 4-for-1)

**Example:**
```python
from alphaweave.data.corporate_actions import SplitAction
from datetime import datetime

split = SplitAction(
    symbol="AAPL",
    date=datetime(2020, 8, 31),
    ratio=4.0  # 4-for-1 split
)
```

---

### `alphaweave.data.corporate_actions.DividendAction`

Represents a cash dividend action.

```python
@dataclass
class DividendAction:
    symbol: str
    date: datetime
    amount: float  # dividend per share
```

**Attributes:**
- `symbol` (str): Stock symbol
- `date` (datetime): Dividend payment date
- `amount` (float): Dividend amount per share

**Example:**
```python
from alphaweave.data.corporate_actions import DividendAction
from datetime import datetime

dividend = DividendAction(
    symbol="AAPL",
    date=datetime(2023, 11, 16),
    amount=0.24  # $0.24 per share
)
```

---

### `alphaweave.data.corporate_actions.CorporateActionsStore`

Stores and provides access to corporate actions by symbol and date.

```python
class CorporateActionsStore:
    def add_split(self, split: SplitAction) -> None
    def add_dividend(self, dividend: DividendAction) -> None
    def get_splits_on_date(self, symbol: str, date: datetime) -> List[SplitAction]
    def get_dividends_on_date(self, symbol: str, date: datetime) -> List[DividendAction]
    def has_actions_for_symbol(self, symbol: str) -> bool
```

**Methods:**

- `add_split(split: SplitAction) -> None`: Add a split action to the store
- `add_dividend(dividend: DividendAction) -> None`: Add a dividend action to the store
- `get_splits_on_date(symbol: str, date: datetime) -> List[SplitAction]`: Get all splits for a symbol on a specific date
- `get_dividends_on_date(symbol: str, date: datetime) -> List[DividendAction]`: Get all dividends for a symbol on a specific date
- `has_actions_for_symbol(symbol: str) -> bool`: Check if there are any corporate actions for a symbol

**Example:**
```python
from alphaweave.data.corporate_actions import (
    CorporateActionsStore,
    SplitAction,
    DividendAction,
)
from datetime import datetime

store = CorporateActionsStore()

# Add a split
split = SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0)
store.add_split(split)

# Add a dividend
dividend = DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24)
store.add_dividend(dividend)

# Retrieve actions
splits = store.get_splits_on_date("AAPL", datetime(2020, 8, 31))
dividends = store.get_dividends_on_date("AAPL", datetime(2023, 11, 16))
```

---

### `alphaweave.data.corporate_actions.load_splits_csv`

```python
load_splits_csv(path: str) -> List[SplitAction]
```

Load split actions from a CSV file.

**Expected CSV format:**
```csv
symbol,date,ratio
AAPL,2020-08-31,4.0
MSFT,2003-02-18,2.0
```

**Parameters:**
- `path` (str): Path to CSV file

**Returns:**
- `List[SplitAction]`: List of split actions

**Example:**
```python
from alphaweave.data.corporate_actions import load_splits_csv

splits = load_splits_csv("data/splits.csv")
```

---

### `alphaweave.data.corporate_actions.load_dividends_csv`

```python
load_dividends_csv(path: str) -> List[DividendAction]
```

Load dividend actions from a CSV file.

**Expected CSV format:**
```csv
symbol,date,amount
AAPL,2023-11-16,0.24
MSFT,2023-11-15,0.75
```

**Parameters:**
- `path` (str): Path to CSV file

**Returns:**
- `List[DividendAction]`: List of dividend actions

**Example:**
```python
from alphaweave.data.corporate_actions import load_dividends_csv

dividends = load_dividends_csv("data/dividends.csv")
```

---

### `alphaweave.data.corporate_actions.build_corporate_actions_store`

```python
build_corporate_actions_store(
    splits: Optional[List[SplitAction]] = None,
    dividends: Optional[List[DividendAction]] = None,
) -> CorporateActionsStore
```

Build a CorporateActionsStore from lists of splits and dividends.

**Parameters:**
- `splits` (Optional[List[SplitAction]]): Optional list of split actions
- `dividends` (Optional[List[DividendAction]]): Optional list of dividend actions

**Returns:**
- `CorporateActionsStore`: Store containing all provided actions

**Example:**
```python
from alphaweave.data.corporate_actions import (
    build_corporate_actions_store,
    SplitAction,
    DividendAction,
)
from datetime import datetime

splits = [
    SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0),
]
dividends = [
    DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24),
]

store = build_corporate_actions_store(splits=splits, dividends=dividends)
```

---

### `alphaweave.engine.portfolio.Portfolio.apply_split`

```python
apply_split(symbol: str, ratio: float) -> None
```

Apply a stock split to a position. Adjusts position size and cost basis per share so that total position cost remains unchanged.

**Parameters:**
- `symbol` (str): Symbol of the position to split
- `ratio` (float): Split ratio (e.g., 2.0 for a 2-for-1 split)

**Behavior:**
- Multiplies position size by ratio
- Divides cost basis per share by ratio
- Keeps total position cost unchanged

**Example:**
```python
from alphaweave.engine.portfolio import Portfolio

portfolio = Portfolio(starting_cash=10000.0)
# ... after buying 100 shares at $50 ...

# Apply 2-for-1 split
portfolio.apply_split("AAPL", ratio=2.0)
# Now: 200 shares at $25 avg_price (total cost unchanged)
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

##### `log_metric(name: str, value: float) -> None`

Log a custom scalar metric. If the engine was provided with a `monitor`, the metric is forwarded (with the current timestamp) via `Monitor.on_metric`. If no monitor is attached, the call is a no-op.

Useful for tracking strategy-specific diagnostics like signal strength or risk estimates that later appear in dashboards.

---

##### `get_state() -> Mapping[str, Any]`

Override to expose strategy-specific serializable state used for live persistence/restarts. Default returns `{}`.

##### `set_state(state: Mapping[str, Any]) -> None`

Inverse of `get_state`. Called by live runners when restoring from persisted state. Default implementation is a no-op.

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
#### Monitoring Parameters

`VectorBacktester.run()` now accepts:

- `monitor` (`Optional[Monitor]`): Hook that receives `on_run_start`, per-bar `BarSnapshot`, per-trade `TradeRecord`, arbitrary metrics, and `on_run_end`.
- `monitor_meta` (`Optional[Mapping[str, Any]]`): Metadata merged into the dictionary passed to `monitor.on_run_start`. By default the backtester reports `mode="backtest"`, `performance_mode`, `trade_at_next_open`, and `capital`.

When `monitor` is supplied, `Strategy.log_metric()` will forward custom metrics to the same monitor.

---

### `alphaweave.live.LiveEngine`

Simple facade that reuses `VectorBacktester` for live/replay workflows while tagging monitor metadata with `mode="live"`.

```python
from alphaweave.live.engine import LiveEngine

engine = LiveEngine(monitor=my_monitor)
result = engine.run(MyStrategy, data={"AAPL": frame}, capital=10_000.0)
```

Arguments mirror `VectorBacktester.run`, and you can still override `monitor`/`monitor_meta` per call.

- **No slippage**: Orders execute exactly at the close price
- **No fees**: No commission or transaction costs
- **No partial fills**: Orders execute fully or not at all
- **No margin/leverage**: Cash-only trading
- **Single asset support**: Multi-asset support is basic (all frames must have the same index)

---

#### Methods

##### `run(strategy_cls, data, capital=100_000.0, ..., corporate_actions=None) -> BacktestResult`

Run the backtest and return results.

**Parameters:**
- `strategy_cls` (Type[Strategy]): Strategy class to instantiate
- `data` (Dict[str, Frame]): Dictionary mapping symbol names to Frame objects
- `capital` (float): Starting capital (default: 100,000.0)
- `fees` (Optional[FeesModel]): Fees model (default: NoFees)
- `slippage` (Optional[SlippageModel]): Slippage model (default: NoSlippage)
- `strategy_kwargs` (Optional[Dict[str, Any]]): Keyword arguments to pass to strategy constructor
- `volume_limit` (Optional[VolumeLimitModel]): Volume limit model (default: VolumeLimitModel)
- `risk_limits` (Optional[RiskLimits]): Risk limits (default: RiskLimits)
- `corporate_actions` (Optional[CorporateActionsStore]): Corporate actions store for splits and dividends (default: None)

**Returns:**
- `BacktestResult`: Results containing equity series and trades

**Corporate Actions:**
When `corporate_actions` is provided, the backtester will:
- Apply stock splits before processing orders on each bar (adjusts position size and cost basis)
- Apply cash dividends before processing orders on each bar (credits cash to portfolio)
- Ensure portfolio value remains consistent across splits

**Example:**
```python
from alphaweave.engine.vector import VectorBacktester
from alphaweave.core.frame import Frame
from alphaweave.data.corporate_actions import (
    build_corporate_actions_store,
    SplitAction,
    DividendAction,
)
from datetime import datetime

# Prepare data
data = {
    "AAPL": frame_aapl,
    "MSFT": frame_msft
}

# Create corporate actions store
splits = [SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0)]
dividends = [DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24)]
corporate_actions = build_corporate_actions_store(splits=splits, dividends=dividends)

# Run backtest
backtester = VectorBacktester()
result = backtester.run(
    MyStrategy,
    data,
    capital=100000.0,
    corporate_actions=corporate_actions
)

# Access results
print(f"Final equity: ${result.equity_series.iloc[-1]:,.2f}")
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

Container for backtest results with comprehensive metrics and analytics.

#### Constructor

```python
BacktestResult(
    equity_series: List[float],
    trades: List[Any],
    timestamps: Optional[List[Any]] = None
)
```

**Parameters:**
- `equity_series` (List[float]): List of equity values, one per bar
- `trades` (List[Any]): List of trade/fill records (Fill objects)
- `timestamps` (Optional[List[Any]]): Optional timestamps for equity series

---

#### Attributes

- `equity_series` (pd.Series): Equity curve values (indexed by timestamp if provided)
- `trades` (pd.DataFrame): Standardized trades DataFrame with columns:
  - `symbol`, `entry_time`, `exit_time`, `entry_price`, `exit_price`
  - `size`, `pnl`, `pnl_pct`, `duration`, `fees`, `slippage`, `direction`
- `returns` (pd.Series): Returns series (pct_change of equity)

---

#### Properties

##### `final_equity: float`

Final equity value.

##### `total_return: float`

Total return as a decimal (e.g., 0.10 for 10%).

##### `max_drawdown: float`

Maximum drawdown as a decimal.

##### `sharpe(rf_annual=0.0, periods_per_year=252) -> float`

Annualized Sharpe ratio.

---

#### Rolling Metrics

##### `rolling_return(window: str = "63D", freq: Literal["calendar", "bars"] = "calendar") -> pd.Series`

Rolling cumulative return over a lookback window.

**Parameters:**
- `window` (str): Window size (e.g., "63D" for calendar, "63" for bars)
- `freq` (Literal): "calendar" for time-based or "bars" for bar-count

**Returns:**
- `pd.Series`: Series of rolling returns

##### `rolling_vol(window: str = "63D", freq: Literal["calendar", "bars"] = "calendar", annualize: bool = True, trading_days: int = 252) -> pd.Series`

Rolling volatility of returns.

**Parameters:**
- `window` (str): Window size
- `freq` (Literal): "calendar" or "bars"
- `annualize` (bool): If True, annualize the volatility
- `trading_days` (int): Trading days per year for annualization

**Returns:**
- `pd.Series`: Series of rolling volatilities

##### `rolling_sharpe(window: str = "63D", freq: Literal["calendar", "bars"] = "calendar", risk_free_rate: float = 0.0, trading_days: int = 252) -> pd.Series`

Rolling Sharpe ratio over the given window.

**Parameters:**
- `window` (str): Window size
- `freq` (Literal): "calendar" or "bars"
- `risk_free_rate` (float): Annual risk-free rate
- `trading_days` (int): Trading days per year

**Returns:**
- `pd.Series`: Series of rolling Sharpe ratios

##### `rolling_drawdown(window: str = "252D", freq: Literal["calendar", "bars"] = "calendar") -> pd.Series`

Rolling max drawdown.

**Parameters:**
- `window` (str): Window size
- `freq` (Literal): "calendar" or "bars"

**Returns:**
- `pd.Series`: Series of rolling max drawdowns

---

#### Trade Analytics

##### `trade_summary() -> Dict[str, Any]`

Basic summary stats for trades.

**Returns:**
- Dictionary with `n_trades`, `win_rate`, `avg_win`, `avg_loss`, `expectancy`, `max_consecutive_wins`, `max_consecutive_losses`, `median_duration`

##### `trade_distribution() -> Dict[str, Any]`

Return distributions or precomputed quantiles for trade metrics.

**Returns:**
- Dictionary with quantiles for `pnl`, `pnl_pct`, `duration`

##### `trade_analytics() -> TradeAnalytics`

Get TradeAnalytics helper for detailed trade analysis.

**Returns:**
- `TradeAnalytics` instance with methods: `by_symbol()`, `by_month()`, `pnl_curve()`

---

#### Turnover & Cost Attribution

##### `turnover(freq: str = "1M") -> pd.Series`

Portfolio turnover per period.

**Parameters:**
- `freq` (str): Resampling frequency (e.g., "1M", "1D")

**Returns:**
- `pd.Series`: Series indexed by period end timestamp

##### `average_slippage_per_share() -> float`

Average slippage per share across all trades.

##### `slippage_cost_series() -> pd.Series`

Time series of cumulative slippage cost.

##### `fee_cost_series() -> pd.Series`

Time series of cumulative fees/commissions.

---

#### Factor Regression

##### `factor_regression(factor_returns: pd.DataFrame, **kwargs) -> FactorRegressionResult`

Run factor regression on strategy returns.

**Parameters:**
- `factor_returns` (pd.DataFrame): DataFrame with factor returns (columns = factors, index = datetime)
- `**kwargs`: Additional arguments for factor_regression

**Returns:**
- `FactorRegressionResult` with alpha, betas, R², t-stats, etc.

**Example:**
```python
factor_returns = pd.DataFrame({
    "SPY": spy_returns,
    "MKT": market_returns,
}, index=dates)

factor_result = result.factor_regression(factor_returns)
print(f"Alpha: {factor_result.alpha:.4f}")
print(f"Beta (SPY): {factor_result.betas['SPY']:.4f}")
print(f"R²: {factor_result.r2:.4f}")
```

---

#### Plotting

##### `plot_equity() -> None`

Plot the equity curve using matplotlib.

**Example:**
```python
result = backtester.run(MyStrategy, data)
result.plot_equity()  # Opens matplotlib window
```

---

## Analysis

### `alphaweave.analysis.factors`

Factor regression and decomposition.

#### `factor_regression(strategy_returns: pd.Series, factor_returns: pd.DataFrame, *, add_constant: bool = True) -> FactorRegressionResult`

Run a simple OLS regression of strategy returns on factor returns.

**Parameters:**
- `strategy_returns` (pd.Series): Strategy returns series (indexed by datetime)
- `factor_returns` (pd.DataFrame): Factor returns DataFrame (columns = factor names, index = datetime)
- `add_constant` (bool): If True, add intercept term (alpha)

**Returns:**
- `FactorRegressionResult` with alpha, betas, R², t-stats, residual_std, n_obs

#### `FactorRegressionResult`

```python
@dataclass
class FactorRegressionResult:
    alpha: float
    betas: pd.Series         # index = factor names
    residual_std: float
    r2: float
    tstats: pd.Series        # index = ["alpha"] + factor names
    n_obs: int
```

---

## Report Generation

### `alphaweave.results.report`

Generate HTML and Markdown reports for backtest results.

#### `generate_markdown_report(result: BacktestResult, *, title: str = "Backtest Report", benchmark: Optional[pd.Series] = None, factor_returns: Optional[pd.DataFrame] = None) -> str`

Generate a Markdown report summarizing the backtest.

**Parameters:**
- `result` (BacktestResult): BacktestResult to report on
- `title` (str): Report title
- `benchmark` (Optional[pd.Series]): Optional benchmark returns for comparison
- `factor_returns` (Optional[pd.DataFrame]): Optional factor returns for regression

**Returns:**
- Markdown string

#### `generate_html_report(result: BacktestResult, *, title: str = "Backtest Report", benchmark: Optional[pd.Series] = None, factor_returns: Optional[pd.DataFrame] = None, include_plots: bool = True) -> str`

Generate an HTML report with optional embedded plots.

**Parameters:**
- `result` (BacktestResult): BacktestResult to report on
- `title` (str): Report title
- `benchmark` (Optional[pd.Series]): Optional benchmark returns for comparison
- `factor_returns` (Optional[pd.DataFrame]): Optional factor returns for regression
- `include_plots` (bool): If True, embed matplotlib plots as base64 images

**Returns:**
- HTML string

**Example:**
```python
from alphaweave.results.report import generate_html_report

html = generate_html_report(
    result,
    title="My Strategy Backtest",
    factor_returns=factor_returns,
    include_plots=True,
)

with open("report.html", "w") as f:
    f.write(html)
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
- **Order execution**: Orders execute at the close price of the current bar. Fees and slippage can be configured via optional parameters.
- **Portfolio tracking**: Positions are tracked with average entry price using weighted averages when adding to existing positions.
- **Corporate actions**: Stock splits preserve portfolio value by adjusting position size and cost basis. Cash dividends are credited to portfolio cash.

---

---

## Portfolio Optimization

### `alphaweave.portfolio.optimizers`

Portfolio optimization algorithms for constructing optimal asset allocations.

#### Cost & Turnover Helpers

- `alphaweave.portfolio.costs.TransactionCostModel`: base interface with `estimate_cost(current_weights, target_weights, prices=None, volumes=None) -> float`.
- Concrete models:
  - `ProportionalCostModel(cost_per_dollar)` — linear turnover cost (scalar or per-asset series).
  - `SpreadBasedCostModel(spread_bps, participation_rate=1.0)` — models half-spread * turnover.
  - `ImpactCostModel(adv, impact_coeff=0.1)` — square-root market impact using average daily volume.
  - `CompositeCostModel(components=list[TransactionCostModel])` — sums multiple models.
- `alphaweave.portfolio.turnover`:
  - `compute_turnover(prev, target)` — sum of absolute weight changes.
  - `TurnoverConstraint(max_turnover=None, max_change_per_asset=None)` — hard caps.
  - `RebalancePenalty(lambda_rebalance=1.0)` — soft penalty that shrinks weights back toward `prev_weights`.

All optimizers accept these objects via optional keyword arguments so that realism is opt-in and backward compatible.

#### `equal_weight(assets: Sequence[str], *, constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult`

Equal weight portfolio optimizer.

**Parameters:**
- `assets` (Sequence[str]): List of asset identifiers
- `constraints` (Optional[PortfolioConstraints]): Optional portfolio constraints

**Returns:**
- `OptimizationResult` with equal weights

**Example:**
```python
from alphaweave.portfolio.optimizers import equal_weight

result = equal_weight(["AAPL", "MSFT", "GOOG"])
weights = result.weights  # Series with equal weights (1/3 each)
```

---

#### `mean_variance(expected_returns: Weights, cov_matrix: pd.DataFrame, *, risk_free_rate: float = 0.0, max_weight: Optional[float] = None, constraints: Optional[PortfolioConstraints] = None, prev_weights: Optional[pd.Series] = None, transaction_cost_model: Optional[TransactionCostModel] = None, turnover_constraint: Optional[TurnoverConstraint] = None, rebalance_penalty: Optional[RebalancePenalty] = None, prices: Optional[pd.Series] = None, volumes: Optional[pd.Series] = None) -> OptimizationResult`

Mean-variance optimizer (maximize Sharpe ratio).

**Parameters:**
- `expected_returns` (Weights): Expected returns per asset.
- `cov_matrix` (pd.DataFrame): Covariance matrix (aligned to expected_returns).
- `risk_free_rate` (float): Risk-free rate for Sharpe calculation.
- `max_weight` (Optional[float]): Maximum weight per asset.
- `constraints` (Optional[PortfolioConstraints]): Optional portfolio constraints.
- `prev_weights` (Optional[pd.Series]): Previous rebalance weights (default zeros). Required when using costs/turnover controls.
- `transaction_cost_model` (Optional[TransactionCostModel]): Estimated cost penalty incorporated directly into the optimization objective.
- `turnover_constraint` (Optional[TurnoverConstraint]): Hard caps on total turnover and/or per-asset change.
- `rebalance_penalty` (Optional[RebalancePenalty]): Soft shrinkage back toward `prev_weights`.
- `prices` / `volumes` (Optional[pd.Series]): Optional market data passed through to the cost model.

**Returns:**
- `OptimizationResult` with optimized weights

**Example:**
```python
from alphaweave.portfolio.optimizers import mean_variance

expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08, "GOOG": 0.06})
cov_matrix = pd.DataFrame({...})  # Covariance matrix

result = mean_variance(expected_returns, cov_matrix)
weights = result.weights  # Optimized weights maximizing Sharpe
```

---

#### `min_variance(cov_matrix: Optional[pd.DataFrame] = None, *, risk_model: Optional[RiskModel] = None, max_weight: Optional[float] = None, constraints: Optional[PortfolioConstraints] = None, prev_weights: Optional[pd.Series] = None, transaction_cost_model: Optional[TransactionCostModel] = None, turnover_constraint: Optional[TurnoverConstraint] = None, rebalance_penalty: Optional[RebalancePenalty] = None, prices: Optional[pd.Series] = None, volumes: Optional[pd.Series] = None) -> OptimizationResult`

Minimum variance optimizer.

**Parameters:**
- `cov_matrix` / `risk_model`: Provide either a covariance matrix or a `RiskModel` supplying one.
- `max_weight`: Maximum weight per asset.
- `constraints`: Optional portfolio constraints.
- `prev_weights`, `transaction_cost_model`, `turnover_constraint`, `rebalance_penalty`, `prices`, `volumes`: same semantics as `mean_variance`.

**Returns:**
- `OptimizationResult` with minimum variance weights

---

#### `risk_parity(cov_matrix: Optional[pd.DataFrame] = None, *, risk_model: Optional[RiskModel] = None, constraints: Optional[PortfolioConstraints] = None, prev_weights: Optional[pd.Series] = None, transaction_cost_model: Optional[TransactionCostModel] = None, turnover_constraint: Optional[TurnoverConstraint] = None, rebalance_penalty: Optional[RebalancePenalty] = None, prices: Optional[pd.Series] = None, volumes: Optional[pd.Series] = None) -> OptimizationResult`

Risk parity optimizer (equal risk contribution).

**Parameters:**
- `cov_matrix` / `risk_model`: Covariance input for risk contributions.
- `constraints`: Optional portfolio constraints.
- `prev_weights`, `transaction_cost_model`, `turnover_constraint`, `rebalance_penalty`, `prices`, `volumes`: optional realism knobs identical to `mean_variance`.

**Returns:**
- `OptimizationResult` with risk parity weights

---

#### `target_volatility(base_weights: Weights, cov_matrix: pd.DataFrame, *, target_vol: float, max_leverage: Optional[float] = None) -> OptimizationResult`

Target volatility allocator.

**Parameters:**
- `base_weights` (Weights): Base portfolio weights
- `cov_matrix` (pd.DataFrame): Covariance matrix
- `target_vol` (float): Target annualized volatility (e.g., 0.10 for 10%)
- `max_leverage` (Optional[float]): Maximum leverage

**Returns:**
- `OptimizationResult` with scaled weights

---

### `alphaweave.portfolio.risk`

Risk estimation helpers for portfolio optimization.

#### `estimate_covariance(returns: pd.DataFrame, *, method: str = "sample", span: int = 60) -> pd.DataFrame`

Estimate asset return covariance matrix.

**Parameters:**
- `returns` (pd.DataFrame): DataFrame with columns = assets, index = dates
- `method` (str): Estimation method ("sample" or "ewma")
- `span` (int): Span parameter for EWMA

**Returns:**
- Covariance matrix DataFrame

#### `estimate_volatility(returns: pd.DataFrame, *, method: str = "sample", span: int = 60, trading_days: int = 252) -> pd.Series`

Per-asset volatility estimate.

**Parameters:**
- `returns` (pd.DataFrame): DataFrame with columns = assets, index = dates
- `method` (str): Estimation method ("sample" or "ewma")
- `span` (int): Span parameter for EWMA
- `trading_days` (int): Trading days per year for annualization

**Returns:**
- Series of annualized volatilities

---

### `alphaweave.portfolio.universe`

Universe selection and ranking utilities.

#### `top_n_by_score(scores: pd.Series, n: int, *, ascending: bool = False) -> Sequence[str]`

Select top-N assets by score.

**Parameters:**
- `scores` (pd.Series): Series indexed by asset
- `n` (int): Number of top assets to select
- `ascending` (bool): If True, select bottom-N

**Returns:**
- List of asset identifiers

#### `normalize_scores_to_weights(scores: pd.Series, *, long_only: bool = True) -> pd.Series`

Map arbitrary scores to portfolio weights.

**Parameters:**
- `scores` (pd.Series): Series indexed by asset with arbitrary scores
- `long_only` (bool): If True, set negative scores to 0 before normalizing

**Returns:**
- Series of normalized weights (sums to 1.0)

---

### `alphaweave.portfolio.constraints`

Portfolio constraint definitions.

#### `PortfolioConstraints`

```python
@dataclass
class PortfolioConstraints:
    weight_bounds: Optional[Mapping[str, WeightBounds]] = None
    default_bounds: Optional[WeightBounds] = None
    target_gross: Optional[float] = 1.0
    long_only: bool = True
```

**Attributes:**
- `weight_bounds`: Per-asset weight bounds
- `default_bounds`: Fallback bounds for assets not in weight_bounds
- `target_gross`: Target gross exposure (sum of absolute weights)
- `long_only`: If True, enforce long-only constraint

#### `WeightBounds`

```python
@dataclass
class WeightBounds:
    lower: float  # e.g. 0.0 for long-only
    upper: float  # e.g. 0.1 for 10% cap
```

---

## Monitoring & Dashboards

### `alphaweave.monitoring.core`

- `BarSnapshot`: dataclass capturing bar-level state (`timestamp`, `prices`, `equity`, `cash`, `positions`, `pnl`, `drawdown`, etc.).
- `TradeRecord`: dataclass for executed fills (`symbol`, `side`, `price`, `size`, `fees`, `slippage`, ...).
- `Monitor`: protocol defining `on_run_start()`, `on_bar()`, `on_trade()`, `on_metric()`, and `on_run_end()`.
- `InMemoryMonitor`: default implementation that stores bars/trades/metrics in memory and exposes `bars_df()`, `trades_df()`, `metrics_df()`.

### `alphaweave.monitoring.run.RunMonitor`

Wrap an `InMemoryMonitor` and gain derived analytics:

- `equity_curve()` / `drawdown_curve()`
- `exposure_over_time()`
- `turnover_over_time()`
- `cost_over_time()`

### `alphaweave.monitoring.plots`

Matplotlib helpers returning `Figure` objects:

- `plot_equity_and_drawdown(run)`
- `plot_exposure_heatmap(run)`
- `plot_turnover(run)`
- `plot_trade_pnl_histogram(run)`

### `alphaweave.monitoring.dashboard.generate_html_dashboard`

Produce a self-contained HTML dashboard embedding the plots above, overview stats, a trades table, and (optionally) live-vs-backtest drift comparisons.

### `alphaweave.analysis.compute_live_drift_series`

Align backtest equity with a `RunMonitor`'s live equity and return `(live - backtest) / backtest` for drift monitoring.

---

## Live Trading Stack

### `alphaweave.live.broker`

- `BrokerFill` / `BrokerAccountState` dataclasses describe fills and account snapshots.
- `Broker` protocol defines the minimal interface (`submit_order`, `cancel_order`, `get_open_orders`, `get_account_state`, `poll_fills`).

### `alphaweave.live.adapters`

- `BrokerConfig`: normalized config payload for adapters.
- Concrete adapters:
  - `MockBrokerAdapter` – instant fills using the backtester portfolio model.
  - `PaperBrokerAdapter` – thin wrapper over the mock adapter with friendlier config defaults.
  - `IBKRAdapter`, `AlpacaAdapter`, `BinanceAdapter` skeletons (raise `NotImplementedError` until wired to real APIs).

### `alphaweave.live.config`

- `LiveConfig`: dataclass containing broker/strategy/datafeed/risk/monitor/persistence sections.
- `load_live_config(path)` reads YAML or JSON (PyYAML optional). Use `parse_live_config(mapping)` if you already have a dict.

### `alphaweave.live.runner.LiveRunner`

High-level orchestrator:

```python
from alphaweave.live.config import load_live_config
from alphaweave.live.runner import LiveRunner

cfg = load_live_config("examples/live_config_template.yaml")
runner = LiveRunner.from_config(cfg)
run_monitor = runner.run()
```

- Builds the broker adapter, replay datafeed, strategy (via dotted path), `LiveEngine`, and `InMemoryMonitor`.
- Writes dashboards (`monitor.dashboard_html`) and persists broker snapshots (`persistence.state_path`) after each run.
- Provides a `RunMonitor` for downstream analysis.

### `alphaweave.live.state`

- `LiveState` dataclass plus `save_live_state(path, state)` / `load_live_state(path)` helpers.
- Strategies can override `get_state/set_state` to persist custom fields.

### `alphaweave.live.datafeed`

- `ReplayDataFeed`: convenience wrapper over static Frames; used by LiveRunner for deterministic dry runs/replays.

---

## Version

This documentation corresponds to **alphaweave version 0.0.1** (Sprints 0-18).

