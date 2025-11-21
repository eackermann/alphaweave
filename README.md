# AlphaWeave

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
- **Vector Backtester**: Bar-by-bar backtesting engine with realistic execution
- **Data loaders**: CSV and Parquet file loaders with automatic column normalization
- **Corporate Actions**: Support for stock splits and cash dividends
- **Technical Indicators**: Built-in indicators (SMA, EMA, RSI, ROC, ATR)
- **Signals**: Signal generation helpers (crossovers, comparisons)
- **Execution Models**: Fees, slippage, and volume limit models
- **Risk Management**: Risk limits and portfolio constraints
- **Analysis Tools**: Walk-forward analysis, robustness testing, parameter sweeps
- **Advanced Metrics**: Rolling Sharpe, drawdown, volatility; trade analytics; factor regression
- **Report Generation**: HTML/Markdown reports with embedded plots and comprehensive analysis
- **Portfolio Optimization**: Mean-variance, risk parity, minimum variance, target volatility optimizers

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

## Features

### Corporate Actions

alphaweave supports stock splits and cash dividends:

```python
from alphaweave.data.corporate_actions import (
    build_corporate_actions_store,
    SplitAction,
    DividendAction,
)
from datetime import datetime

# Create corporate actions
splits = [
    SplitAction(symbol="AAPL", date=datetime(2020, 8, 31), ratio=4.0),  # 4-for-1 split
]
dividends = [
    DividendAction(symbol="AAPL", date=datetime(2023, 11, 16), amount=0.24),  # $0.24/share
]

store = build_corporate_actions_store(splits=splits, dividends=dividends)

# Use in backtest
result = aw.engine.vector.VectorBacktester().run(
    BuyAndHold,
    data={"AAPL": frame},
    capital=10000.0,
    corporate_actions=store  # Splits preserve equity, dividends add cash
)
```

### Technical Indicators

Built-in indicators for strategy development:

```python
class MyStrategy(aw.strategy.base.Strategy):
    def init(self):
        self.sma_period = 20
        self.rsi_period = 14
    
    def next(self, i):
        sma = self.sma("SYMBOL", self.sma_period)
        rsi = self.rsi("SYMBOL", self.rsi_period)
        close = self.close("SYMBOL")
        
        # Trading logic using indicators
        if close > sma and rsi < 70:
            self.order_target_percent("SYMBOL", 1.0)
```

### Execution Models

Configure fees, slippage, and volume limits:

```python
from alphaweave.execution.fees import FixedFees
from alphaweave.execution.slippage import PercentageSlippage

fees = FixedFees(commission=1.0)  # $1 per trade
slippage = PercentageSlippage(rate=0.001)  # 0.1% slippage

result = backtester.run(
    MyStrategy,
    data={"AAPL": frame},
    fees=fees,
    slippage=slippage
)
```

### Advanced Metrics & Reporting

Analyze backtest results with comprehensive metrics and generate reports:

```python
from alphaweave.results.report import generate_html_report
import pandas as pd

result = backtester.run(MyStrategy, data={"AAPL": frame})

# Rolling metrics
rolling_sharpe = result.rolling_sharpe("63D")  # 63-day rolling Sharpe
rolling_dd = result.rolling_drawdown("252D")    # 252-day rolling drawdown

# Trade analytics
summary = result.trade_summary()
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Expectancy: ${summary['expectancy']:.2f}")

# Factor regression
factor_returns = pd.DataFrame({
    "SPY": spy_returns,
}, index=dates)
factor_result = result.factor_regression(factor_returns)
print(f"Alpha: {factor_result.alpha:.4f}")
print(f"Beta: {factor_result.betas['SPY']:.4f}")

# Generate HTML report
html = generate_html_report(
    result,
    title="My Strategy Backtest",
    factor_returns=factor_returns,
    include_plots=True,
)
with open("report.html", "w") as f:
    f.write(html)
```

### Portfolio Optimization

Build optimized portfolios using standard optimizers:

```python
from alphaweave.portfolio.optimizers import risk_parity, min_variance, target_volatility
from alphaweave.portfolio.risk import estimate_covariance
from alphaweave.portfolio.universe import top_n_by_score, normalize_scores_to_weights

class OptimizedPortfolio(Strategy):
    def init(self):
        self.assets = ["SPY", "TLT", "GLD", "QQQ"]
        self.lookback = 60
    
    def next(self, i):
        if not (self.schedule.every("1M") and self.schedule.at_close()):
            return
        
        # Get historical returns
        returns_df = self._get_recent_returns(self.assets, self.lookback)
        
        # Estimate covariance
        cov = estimate_covariance(returns_df, method="ewma")
        
        # Optimize: risk parity
        result = risk_parity(cov)
        
        # Apply weights
        for symbol, w in result.weights.items():
            self.order_target_percent(symbol, w)
```

## Documentation

- **[API.md](API.md)**: Complete API reference
- **[context.md](context.md)**: Comprehensive codebase context and architecture
- **Examples**: See `examples/` directory for strategy examples

## License

See LICENSE file for details.

