# alphaweave Context Documentation

**Version:** 0.0.1  
**Python:** 3.11+  
**Purpose:** Comprehensive context for understanding the alphaweave backtesting framework

This document provides a complete overview of the alphaweave codebase structure, implementation details, design decisions, and usage patterns. Use this as a reference when working with or extending the framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Core Architecture](#core-architecture)
4. [Module Details](#module-details)
5. [Key Design Decisions](#key-design-decisions)
6. [Usage Patterns](#usage-patterns)
7. [Extension Points](#extension-points)

---

## Overview

alphaweave is a Python backtesting framework designed for quantitative trading strategy development and testing. It provides:

- **Unified data interface**: Works seamlessly with both Pandas and Polars DataFrames
- **Simple strategy API**: Minimal base class for implementing trading strategies
- **Realistic execution**: Fees, slippage, volume limits, and risk management
- **Corporate actions**: Stock splits and cash dividends support
- **Technical analysis**: Built-in indicators and signal generators
- **Analysis tools**: Walk-forward analysis, robustness testing, parameter optimization

### Philosophy

- **Simplicity**: Minimal API surface, easy to understand and extend
- **Flexibility**: Support multiple data backends (Pandas/Polars)
- **Realism**: Account for execution costs and market constraints
- **Type safety**: Comprehensive type hints throughout
- **Testability**: Well-tested with comprehensive test suite

---

## Repository Structure

```
alphaweave/
├── alphaweave/              # Main package
│   ├── __init__.py          # Package initialization, public API exports
│   ├── core/                # Core abstractions
│   │   ├── frame.py         # Frame abstraction (Pandas/Polars wrapper)
│   │   └── types.py         # Core dataclasses (Bar, Order, Fill, Position)
│   ├── data/                # Data loading and management
│   │   ├── loaders.py       # CSV/Parquet loaders with normalization
│   │   ├── timeframes.py    # Timeframe resampling utilities
│   │   └── corporate_actions.py  # Splits and dividends support
│   ├── strategy/            # Strategy framework
│   │   └── base.py          # Strategy base class
│   ├── indicators/          # Technical indicators
│   │   ├── base.py          # Base indicator class
│   │   ├── sma.py           # Simple Moving Average
│   │   ├── ema.py           # Exponential Moving Average
│   │   ├── rsi.py           # Relative Strength Index
│   │   ├── roc.py           # Rate of Change
│   │   └── atr.py           # Average True Range
│   ├── signals/             # Signal generation
│   │   ├── base.py          # Base signal class
│   │   ├── crossover.py     # Crossover/crossunder signals
│   │   └── comparison.py   # Comparison signals (>, <, ==)
│   ├── engine/              # Backtesting engine
│   │   ├── base.py          # BaseBacktester protocol
│   │   ├── vector.py        # VectorBacktester implementation
│   │   ├── portfolio.py    # Portfolio accounting
│   │   └── risk.py          # Risk limits and constraints
│   ├── execution/           # Execution models
│   │   ├── fees.py          # Fee models (commission, spread)
│   │   ├── slippage.py      # Slippage models
│   │   └── volume.py        # Volume limit models
│   ├── results/             # Results and metrics
│   │   ├── result.py        # BacktestResult with metrics
│   │   ├── report.py        # HTML/Markdown report generation
│   │   └── trade_analytics.py  # Trade analytics helper
│   ├── analysis/            # Analysis tools
│   │   ├── walkforward.py   # Walk-forward analysis
│   │   ├── robustness.py   # Robustness testing
│   │   └── factors.py       # Factor regression and decomposition
│   ├── portfolio/           # Portfolio construction and optimization
│   │   ├── optimizers.py    # Portfolio optimizers (mean-variance, risk parity, etc.)
│   │   ├── constraints.py  # Portfolio constraint definitions
│   │   ├── risk.py          # Risk estimation helpers
│   │   └── universe.py      # Universe selection and ranking utilities
│   └── utils/               # Utilities
│       ├── time.py          # Time utilities
│       ├── patterns.py      # Pattern matching
│       └── download_price_data.py  # Data download helpers
├── tests/                   # Test suite
│   ├── test_*.py            # Unit and integration tests
│   └── conftest.py          # Pytest configuration
├── examples/                # Example strategies
│   ├── buy_and_hold.py
│   ├── sma_crossover.py
│   ├── rsi_reversion.py
│   └── ...
├── README.md                # User-facing documentation
├── API.md                   # Complete API reference
├── context.md               # This file
└── pyproject.toml           # Project configuration

```

---

## Core Architecture

### Data Flow

```
CSV/Parquet Files
    ↓
load_csv() / load_parquet()
    ↓
Frame (unified Pandas/Polars interface)
    ↓
Strategy (accesses data via Frame)
    ↓
VectorBacktester.run()
    ↓
BacktestResult (equity series, trades, metrics)
```

### Execution Flow

```
1. Data Loading
   - Load CSV/Parquet files
   - Normalize column names
   - Create Frame objects

2. Strategy Initialization
   - Instantiate strategy with data
   - Call strategy.init()

3. Bar-by-Bar Loop
   For each bar:
     a. Get current prices
     b. Apply corporate actions (splits, dividends)
     c. Call strategy.next(i)
     d. Collect orders from strategy
     e. Process orders:
        - Apply risk limits
        - Apply volume limits
        - Calculate slippage
        - Execute fills
        - Apply fees
     f. Update portfolio
     g. Record equity value

4. Return Results
   - BacktestResult with equity series and trades
   - Calculate metrics (Sharpe, drawdown, etc.)
```

### Key Abstractions

#### Frame
Unified interface for time-series OHLCV data, abstracting over Pandas and Polars:
- `Frame.from_pandas(df)` - Create from Pandas DataFrame
- `Frame.from_polars(pl_df)` - Create from Polars DataFrame
- `frame.to_pandas()` - Convert to Pandas
- `frame.to_polars()` - Convert to Polars

#### Strategy
Base class for trading strategies:
- `init()` - Called once before backtest
- `next(i)` - Called each bar, implement trading logic here
- `order_target_percent(symbol, target)` - Place orders
- Helper methods: `close()`, `sma()`, `ema()`, `rsi()`, etc.

#### Portfolio
Tracks cash and positions:
- `cash` - Available cash
- `positions` - Dict[str, Position] of current positions
- `apply_fill(fill)` - Update portfolio with trade
- `apply_split(symbol, ratio)` - Apply stock split
- `total_value(prices)` - Calculate total portfolio value

#### VectorBacktester
Main backtesting engine:
- `run(strategy_cls, data, ...)` - Execute backtest
- Handles order execution, portfolio updates, corporate actions
- Supports fees, slippage, volume limits, risk limits

---

## Module Details

### `alphaweave.core`

**Purpose:** Core abstractions and types

**Key Components:**
- `Frame`: Unified DataFrame interface
- `Bar`: OHLCV bar dataclass
- `Order`: Order request dataclass
- `Fill`: Executed trade dataclass
- `Position`: Portfolio position dataclass

**Design Notes:**
- Frame uses composition to wrap Pandas/Polars DataFrames
- All core types are dataclasses for immutability and clarity
- Type hints throughout for IDE support

### `alphaweave.data`

**Purpose:** Data loading and management

**Key Components:**
- `load_csv()`: Load CSV with automatic column normalization
- `load_parquet()`: Load Parquet files
- `load_directory()`: Load multiple files from directory
- `resample_frame()`: Resample to different timeframes
- `corporate_actions.py`: Split and dividend support

**Design Notes:**
- Automatic column name normalization (case-insensitive, handles variations)
- Supports adjusted close prices
- Corporate actions loaded from CSV or programmatically

### `alphaweave.strategy`

**Purpose:** Strategy framework

**Key Components:**
- `Strategy`: Base class for all strategies
- Helper methods for common operations (indicators, signals, data access)

**Design Notes:**
- Minimal API: only `init()` and `next()` required
- Helper methods provide convenient access to indicators
- Supports both single Frame and multi-symbol dict

### `alphaweave.indicators`

**Purpose:** Technical indicators

**Key Components:**
- `SMA`: Simple Moving Average
- `EMA`: Exponential Moving Average
- `RSI`: Relative Strength Index
- `ROC`: Rate of Change
- `ATR`: Average True Range

**Design Notes:**
- Lazy evaluation: indicators computed on-demand
- Caching: results cached to avoid recomputation
- Works with Frame, Series, or arrays

### `alphaweave.signals`

**Purpose:** Signal generation

**Key Components:**
- `crossover()`: Detect when series crosses above another
- `crossunder()`: Detect when series crosses below another
- `greater_than()`: Comparison signals
- `less_than()`: Comparison signals

**Design Notes:**
- Signals return boolean arrays
- Can be used directly in strategy logic
- Handles NaN values gracefully

### `alphaweave.engine`

**Purpose:** Backtesting engine

**Key Components:**
- `VectorBacktester`: Main backtesting engine
- `Portfolio`: Portfolio accounting
- `RiskLimits`: Risk management constraints

**Design Notes:**
- Bar-by-bar execution model
- Applies corporate actions before processing orders
- Supports multiple order types (market, limit, stop, stop-limit)
- Risk limits: max position size, max gross leverage

### `alphaweave.execution`

**Purpose:** Execution realism

**Key Components:**
- `FeesModel`: Commission and fee models
- `SlippageModel`: Slippage models (fixed, percentage, volume-based)
- `VolumeLimitModel`: Volume-based position sizing limits

**Design Notes:**
- Pluggable models: easy to add custom implementations
- Default models: NoFees, NoSlippage for simplified backtests

### `alphaweave.results`

**Purpose:** Results and metrics

**Key Components:**
- `BacktestResult`: Container for backtest results
- Metrics: Sharpe ratio, max drawdown, total return

**Design Notes:**
- Equity series stored as pandas Series for easy analysis
- Metrics computed as properties for lazy evaluation
- Rolling metrics support both calendar-based and bar-count windows
- Trade analytics provide comprehensive trade-level statistics
- Factor regression enables beta exposure and alpha analysis
- Report generation creates self-contained HTML/Markdown reports

### `alphaweave.analysis`

**Purpose:** Analysis tools

**Key Components:**
- `WalkForwardAnalysis`: Walk-forward optimization
- `RobustnessTest`: Robustness testing utilities
- `factor_regression`: Factor regression and decomposition
- `parameter_sweep`: Parameter optimization with parallel support

**Design Notes:**
- Separate from core engine for modularity
- Can be used independently or with backtester
- Parallel processing support for parameter sweeps (n_jobs parameter)
- Factor regression uses numpy OLS for performance

### `alphaweave.portfolio`

**Purpose:** Portfolio construction and optimization

**Key Components:**
- `optimizers`: Portfolio optimizers (equal weight, mean-variance, min-variance, risk parity, target volatility)
- `constraints`: Portfolio constraint definitions (weight bounds, long-only, leverage caps)
- `risk`: Risk estimation helpers (covariance, volatility)
- `universe`: Universe selection and ranking utilities

**Design Notes:**
- Optimizers are pure functions - no implicit data access
- Strategies build inputs (returns, covariance) and call optimizers
- Optimizers return weights that strategies apply via order_target_percent
- Uses scipy.optimize for constrained optimization
- Supports both sample and EWMA covariance estimation

---

## Key Design Decisions

### 1. Frame Abstraction

**Decision:** Create unified Frame interface over Pandas/Polars

**Rationale:**
- Allows strategies to work with either backend
- Hides implementation details from users
- Makes it easy to switch backends

**Implementation:**
- Frame wraps backend DataFrame
- Provides unified API (to_pandas, to_polars)
- Handles datetime index/column differences

### 2. Strategy API

**Decision:** Minimal base class with only `init()` and `next()`

**Rationale:**
- Simple to understand and implement
- Flexible: strategies can do anything in `next()`
- Helper methods provide convenience without complexity

**Implementation:**
- Base class provides helper methods
- Strategies override `init()` and `next()`
- Orders collected via `_orders` list

### 3. Corporate Actions

**Decision:** Apply splits and dividends before order processing

**Rationale:**
- Ensures portfolio state is correct before strategy decisions
- Splits preserve portfolio value (adjust size and cost basis)
- Dividends credited as cash flows

**Implementation:**
- CorporateActionsStore indexes actions by symbol and date
- Applied in bar loop before strategy.next()
- Splits adjust position size and avg_price
- Dividends credit cash = position_size * dividend_amount

### 4. Execution Models

**Decision:** Pluggable models for fees, slippage, volume

**Rationale:**
- Allows different levels of realism
- Easy to test without costs
- Can add custom models

**Implementation:**
- Protocol-based design
- Default models: NoFees, NoSlippage
- Models can be swapped per backtest

### 5. Type Safety

**Decision:** Comprehensive type hints throughout

**Rationale:**
- Better IDE support
- Catch errors early
- Self-documenting code

**Implementation:**
- All public APIs have type hints
- Uses `typing` module extensively
- Dataclasses for structured data

---

## Usage Patterns

### Basic Strategy

```python
import alphaweave as aw
from alphaweave.strategy.base import Strategy

class MyStrategy(Strategy):
    def init(self):
        # Initialize indicators, variables
        self.sma_period = 20
    
    def next(self, i):
        # Access data
        close = self.close("SYMBOL")
        sma = self.sma("SYMBOL", self.sma_period)
        
        # Trading logic
        if close > sma:
            self.order_target_percent("SYMBOL", 1.0)  # Go long
        else:
            self.order_target_percent("SYMBOL", 0.0)  # Exit
```

### Loading Data

```python
from alphaweave.data.loaders import load_csv, load_directory

# Load single file
frame = load_csv("data/AAPL.csv", symbol="AAPL")

# Load directory
data = load_directory("data/", symbols=["AAPL", "MSFT"])
```

### Running Backtest

```python
from alphaweave.engine.vector import VectorBacktester

backtester = VectorBacktester()
result = backtester.run(
    MyStrategy,
    data={"AAPL": frame_aapl},
    capital=100000.0
)

# Access results
print(f"Final equity: ${result.final_equity:,.2f}")
print(f"Sharpe ratio: {result.sharpe():.2f}")
print(f"Max drawdown: {result.max_drawdown:.2%}")
```

### Corporate Actions

```python
from alphaweave.data.corporate_actions import (
    build_corporate_actions_store,
    SplitAction,
    DividendAction,
)
from datetime import datetime

# Create corporate actions
splits = [
    SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0),
]
dividends = [
    DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24),
]

store = build_corporate_actions_store(splits=splits, dividends=dividends)

# Use in backtest
result = backtester.run(
    MyStrategy,
    data={"AAPL": frame_aapl},
    corporate_actions=store
)
```

### With Execution Models

```python
from alphaweave.execution.fees import FixedFees
from alphaweave.execution.slippage import PercentageSlippage

fees = FixedFees(commission=1.0)  # $1 per trade
slippage = PercentageSlippage(rate=0.001)  # 0.1% slippage

result = backtester.run(
    MyStrategy,
    data={"AAPL": frame_aapl},
    fees=fees,
    slippage=slippage
)
```

### Multi-Symbol Strategy

```python
class MultiAssetStrategy(Strategy):
    def init(self):
        self.weights = {"AAPL": 0.6, "MSFT": 0.4}
    
    def next(self, i):
        # Allocate portfolio across assets
        for symbol, weight in self.weights.items():
            self.order_target_percent(symbol, weight)
```

### Using Indicators

```python
class IndicatorStrategy(Strategy):
    def init(self):
        self.sma_fast = 10
        self.sma_slow = 30
        self.rsi_period = 14
    
    def next(self, i):
        sma_fast = self.sma("SYMBOL", self.sma_fast)
        sma_slow = self.sma("SYMBOL", self.sma_slow)
        rsi = self.rsi("SYMBOL", self.rsi_period)
        
        # Golden cross
        if self.crossover(sma_fast, sma_slow):
            self.order_target_percent("SYMBOL", 1.0)
        
        # RSI oversold
        if rsi < 30:
            self.order_target_percent("SYMBOL", 1.0)
        elif rsi > 70:
            self.order_target_percent("SYMBOL", 0.0)
```

---

## Extension Points

### Adding New Indicators

1. Create indicator class in `alphaweave/indicators/`
2. Inherit from `BaseIndicator`
3. Implement `compute()` method
4. Add helper method to `Strategy` class

### Adding New Execution Models

1. Create model class in `alphaweave/execution/`
2. Implement required methods (see existing models for interface)
3. Use in `VectorBacktester.run()` via parameters

### Adding New Signal Types

1. Create signal function in `alphaweave/signals/`
2. Return boolean array
3. Add helper method to `Strategy` class if needed

### Custom Backtester

1. Implement `BaseBacktester` protocol
2. Implement `run()` method
3. Return `BacktestResult`

---

## Testing

The codebase includes comprehensive tests:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Strategy tests**: Test strategy behavior
- **Execution tests**: Test execution models
- **Corporate actions tests**: Test splits and dividends

Run tests:
```bash
pytest
```

Test coverage includes:
- All core modules
- All indicators
- All signals
- Execution models
- Corporate actions
- Portfolio operations

---

## Dependencies

**Core:**
- `pandas>=2.0` - DataFrame operations
- `numpy>=1.25` - Numerical operations
- `polars>=0.19` - Alternative DataFrame backend

**Optional (dev):**
- `pytest>=8.0` - Testing framework
- `matplotlib>=3.7` - Plotting (for result visualization)

---

## Version History

**0.0.1 (Sprint 0-11):**
- Initial framework
- Frame abstraction
- Strategy API
- Vector backtester
- Data loaders
- Technical indicators
- Signals
- Execution models (fees, slippage, volume)
- Risk management
- Corporate actions (splits, dividends)
- Analysis tools (walk-forward, robustness)
- Multi-resolution execution (intraday with daily signals)
- Event engine & scheduling (weekly/monthly boundaries, earnings events)
- Performance optimizations (fast mode, caching, parallel sweeps)
- Advanced metrics & reporting (rolling metrics, trade analytics, factor regression, HTML reports)
- Portfolio optimization (mean-variance, risk parity, min-variance, target volatility)

---

## Common Patterns

### Pattern 1: Simple Moving Average Crossover

```python
class SMACrossover(Strategy):
    def init(self):
        self.fast = 10
        self.slow = 30
    
    def next(self, i):
        fast_sma = self.sma("SYMBOL", self.fast)
        slow_sma = self.sma("SYMBOL", self.slow)
        
        if self.crossover(fast_sma, slow_sma):
            self.order_target_percent("SYMBOL", 1.0)
        elif self.crossunder(fast_sma, slow_sma):
            self.order_target_percent("SYMBOL", 0.0)
```

### Pattern 2: Mean Reversion with RSI

```python
class RSIReversion(Strategy):
    def init(self):
        self.rsi_period = 14
        self.oversold = 30
        self.overbought = 70
    
    def next(self, i):
        rsi = self.rsi("SYMBOL", self.rsi_period)
        
        if rsi < self.oversold:
            self.order_target_percent("SYMBOL", 1.0)
        elif rsi > self.overbought:
            self.order_target_percent("SYMBOL", 0.0)
```

### Pattern 3: Multi-Timeframe Strategy

```python
class MultiTimeframeStrategy(Strategy):
    def init(self):
        # Access weekly data
        self.weekly_frame = self.data["SYMBOL_WEEKLY"]
    
    def next(self, i):
        # Get weekly trend
        weekly_df = self.weekly_frame.to_pandas()
        weekly_sma = weekly_df["close"].rolling(20).mean()
        
        # Only trade with trend
        if weekly_sma.iloc[i] > weekly_sma.iloc[i-1]:
            # Use daily data for entries
            daily_close = self.close("SYMBOL")
            daily_sma = self.sma("SYMBOL", 10)
            
            if daily_close > daily_sma:
                self.order_target_percent("SYMBOL", 1.0)
```

---

## Best Practices

1. **Data Validation**: Always validate data before backtesting
2. **Error Handling**: Handle missing data gracefully in strategies
3. **Performance**: Use vectorized operations when possible
4. **Testing**: Test strategies on multiple time periods
5. **Documentation**: Document strategy logic and parameters
6. **Risk Management**: Always use risk limits in production
7. **Corporate Actions**: Include splits and dividends for realistic results

---

## Troubleshooting

### Common Issues

**Issue:** "Missing required column"
- **Solution:** Ensure CSV has datetime, open, high, low, close, volume columns (case-insensitive)

**Issue:** "Equity drops on split date"
- **Solution:** Ensure corporate_actions parameter is provided to backtester

**Issue:** "Strategy not placing orders"
- **Solution:** Check that `order_target_percent()` is called in `next()` method

**Issue:** "NaN values in indicators"
- **Solution:** Indicators need enough data points (period length) before producing values

---

## Future Enhancements

Potential areas for extension:
- More indicators (MACD, Bollinger Bands, etc.)
- More order types (trailing stop, bracket orders)
- Portfolio optimization tools
- Real-time data integration
- More execution models
- Additional corporate actions (mergers, spin-offs)

---

## Contact & Contributing

For questions, issues, or contributions, see the main README.md and API.md files.

This context document is maintained alongside the codebase and updated with each major release.

