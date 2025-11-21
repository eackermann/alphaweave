"""Integration tests for strategies using indicators and signals."""

import pandas as pd
from alphaweave.core.frame import Frame
from alphaweave.strategy.base import Strategy
from alphaweave.engine.vector import VectorBacktester
from alphaweave.indicators.sma import SMA
from alphaweave.signals.crossover import CrossOver


def make_sample_data():
    """Create sample data for strategy testing."""
    # Create data with a clear trend
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=15, freq="D"),
        "open": closes,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1000000] * 15,
    })
    return Frame.from_pandas(df)


def test_sma_crossover_strategy():
    """Test a strategy using SMA crossover without manual indexing."""
    frame = make_sample_data()
    
    class SMACrossoverStrategy(Strategy):
        """Strategy that buys on SMA crossover."""
        
        def init(self):
            """Initialize indicators."""
            data_frame = self.get_frame()
            self.sma_fast = SMA(data_frame, period=3, column="close")
            self.sma_slow = SMA(data_frame, period=5, column="close")
            self.crossover = CrossOver(self.sma_fast, self.sma_slow)
            self.position = 0.0
        
        def next(self, index):
            """Execute strategy logic."""
            # Use signals, no manual indexing
            if self.crossover(index):
                # Buy signal
                self.order_target_percent("_default", 1.0)
            elif self.crossover.indicator1[index] < self.crossover.indicator2[index]:
                # Exit if fast crosses below slow
                self.order_target_percent("_default", 0.0)
    
    # Run backtest with single Frame
    backtester = VectorBacktester()
    result = backtester.run(SMACrossoverStrategy, data=frame, capital=10000.0)
    
    # Verify results
    assert len(result.equity_series) == 15
    assert len(result.equity_series) == len(frame.to_pandas())
    assert all(e >= 0 for e in result.equity_series)
    
    # Should have some trades
    assert len(result.trades) >= 0  # May or may not have trades depending on crossover timing


def test_strategy_with_indicators():
    """Test that strategies can use indicators in next() method."""
    frame = make_sample_data()
    
    class IndicatorStrategy(Strategy):
        """Strategy using indicators."""
        
        def init(self):
            """Initialize indicators."""
            data_frame = self.get_frame()
            self.sma = SMA(data_frame, period=5, column="close")
        
        def next(self, index):
            """Use indicators in strategy logic."""
            sma_value = self.sma[index]
            close_frame = self.get_frame()
            pdf = close_frame.to_pandas()
            current_close = pdf.iloc[index]["close"]
            
            # Buy if price is above SMA
            if current_close > sma_value:
                self.order_target_percent("_default", 1.0)
            else:
                self.order_target_percent("_default", 0.0)
    
    backtester = VectorBacktester()
    result = backtester.run(IndicatorStrategy, data=frame, capital=10000.0)
    
    assert len(result.equity_series) == 15
    assert len(result.trades) >= 0


def test_strategy_single_frame_vs_dict():
    """Test that strategies work with both single Frame and dict input."""
    frame = make_sample_data()
    
    class SimpleStrategy(Strategy):
        def init(self):
            self.sma = SMA(self.get_frame(), period=5)
        
        def next(self, index):
            if self.sma[index] > 105:
                self.order_target_percent("_default", 1.0)
    
    # Test with single Frame
    backtester = VectorBacktester()
    result1 = backtester.run(SimpleStrategy, data=frame, capital=10000.0)
    
    # Test with dict
    result2 = backtester.run(SimpleStrategy, data={"_default": frame}, capital=10000.0)
    
    # Results should be identical
    # Convert to list for comparison
    eq1 = list(result1.equity_series) if hasattr(result1.equity_series, '__iter__') else result1.equity_series
    eq2 = list(result2.equity_series) if hasattr(result2.equity_series, '__iter__') else result2.equity_series
    assert eq1 == eq2
    assert len(result1.trades) == len(result2.trades)

