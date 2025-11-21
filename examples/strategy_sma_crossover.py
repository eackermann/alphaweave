"""SMA Crossover Strategy Example.

This example demonstrates a simple moving average crossover strategy using
alphaweave indicators and signals without manual indexing.
"""

import alphaweave as aw
import pandas as pd

# Create sample price data
df = pd.DataFrame({
    "datetime": pd.date_range("2020-01-01", periods=50, freq="D"),
    "open": [100 + i * 0.5 + (i % 3) for i in range(50)],
    "high": [102 + i * 0.5 + (i % 3) for i in range(50)],
    "low": [98 + i * 0.5 + (i % 3) for i in range(50)],
    "close": [100 + i * 0.5 + (i % 3) for i in range(50)],
    "volume": [1000000] * 50,
})

frame = aw.Frame.from_pandas(df)


class SMACrossoverStrategy(aw.strategy.base.Strategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Buys when fast SMA crosses above slow SMA.
    Sells when fast SMA crosses below slow SMA.
    """

    def init(self):
        """Initialize indicators and signals."""
        # Get the data frame
        data_frame = self.get_frame()
        
        # Create two SMAs: fast (5 period) and slow (20 period)
        self.sma_fast = aw.indicators.SMA(data_frame, period=5, column="close")
        self.sma_slow = aw.indicators.SMA(data_frame, period=20, column="close")
        
        # Create crossover signals
        self.buy_signal = aw.signals.CrossOver(self.sma_fast, self.sma_slow)
        self.sell_signal = aw.signals.CrossUnder(self.sma_fast, self.sma_slow)

    def next(self, index):
        """
        Execute strategy logic at each bar.
        
        No manual indexing needed - use signals and indicators directly.
        """
        # Check for buy signal (fast crosses above slow)
        if self.buy_signal(index):
            self.order_target_percent("_default", 1.0)  # Go 100% long
        
        # Check for sell signal (fast crosses below slow)
        elif self.sell_signal(index):
            self.order_target_percent("_default", 0.0)  # Exit position


if __name__ == "__main__":
    # Run backtest
    backtester = aw.engine.vector.VectorBacktester()
    result = backtester.run(
        SMACrossoverStrategy,
        data=frame,  # Single Frame input
        capital=10000.0
    )
    
    # Print results
    print("=" * 60)
    print("SMA Crossover Strategy Results")
    print("=" * 60)
    print(f"Starting Capital: $10,000.00")
    final_equity = float(result.equity_series.iloc[-1])
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {(final_equity / 10000.0 - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(result.trades)}")
    print(f"Number of Bars: {len(result.equity_series)}")
    print()
    
    # Show trade details
    if result.trades:
        print("Trade Details:")
        print("-" * 60)
        for i, trade in enumerate(result.trades[:10], 1):  # Show first 10 trades
            print(
                f"Trade {i}: {trade.symbol} | "
                f"Size: {trade.size:+.2f} | "
                f"Price: ${trade.price:.2f} | "
                f"Date: {trade.datetime.strftime('%Y-%m-%d')}"
            )
        if len(result.trades) > 10:
            print(f"... and {len(result.trades) - 10} more trades")
    
    print()
    print("Equity Curve (first 10 and last 10 values):")
    print("-" * 60)
    eq_list = list(result.equity_series)
    print(f"First 10: {[f'${e:,.2f}' for e in eq_list[:10]]}")
    print(f"Last 10:  {[f'${e:,.2f}' for e in eq_list[-10:]]}")

