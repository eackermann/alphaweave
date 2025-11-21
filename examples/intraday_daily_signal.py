"""Example: Daily signals with intraday execution.

This example demonstrates:
- Daily SMA signal on intraday bars
- Trade at next open mechanics
- VWAP execution price model
- Intraday partial fills with max_participation_rate
"""

import alphaweave as aw
import pandas as pd
from alphaweave.execution.price_models import VWAPPriceModel, OpenClosePriceModel
from alphaweave.execution.volume import VolumeLimitModel


class DailySignalIntradayExec(aw.strategy.base.Strategy):
    """
    Strategy that uses daily signals but executes on intraday bars.
    
    - Generates signals at daily close
    - Executes at next day's open (via trade_at_next_open)
    - Uses VWAP for execution price
    """
    
    def init(self):
        """Initialize strategy."""
        # Register daily timeframe
        self.register_timeframe("daily", "1D", symbols=["TQQQ"])
        
        # Get resampled daily close series
        self.daily_close = self.resampled_close("TQQQ", "1D")
        
        # Calculate daily SMA
        self.daily_sma20 = self.daily_close.rolling(20).mean()
    
    def next(self, i):
        """
        Called on each intraday bar.
        
        Only makes decisions at daily close, but execution happens
        on intraday bars.
        """
        # Check if this is the daily close (last intraday bar of the day)
        if not self.is_session_close("1D"):
            return  # Only decide at daily close
        
        # Get current daily index
        now = self.now()
        if now is None:
            return
        
        # Find the daily index corresponding to current timestamp
        daily_df = self.daily_close.to_frame()
        daily_idx = daily_df.index.get_indexer([now], method="pad")[0]
        
        if daily_idx < 0 or daily_idx >= len(self.daily_close):
            return
        
        # Get daily values
        close_today = self.daily_close.iloc[daily_idx]
        sma_today = self.daily_sma20.iloc[daily_idx]
        
        # Simple SMA crossover strategy
        if pd.notna(close_today) and pd.notna(sma_today):
            if close_today > sma_today:
                # Go long
                self.order_target_percent("TQQQ", 1.0)
            else:
                # Exit position
                self.order_target_percent("TQQQ", 0.0)


def main():
    """Run the example."""
    # Create sample intraday data (5-minute bars)
    # In practice, you'd load this from a data source
    dates = pd.date_range("2021-01-01 09:30", "2021-01-31 16:00", freq="5T")
    
    # Simulate price data with some trend
    n_bars = len(dates)
    base_price = 50.0
    prices = []
    for i in range(n_bars):
        # Add some trend and noise
        price = base_price + (i / 100) + (i % 100) * 0.01
        prices.append(price)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": [p * 1.005 for p in prices],
        "volume": [10000] * n_bars,
    })
    
    frame = aw.core.frame.Frame.from_pandas(df)
    
    # Configure execution models
    price_model = VWAPPriceModel()  # Use VWAP for execution
    volume_model = VolumeLimitModel(max_participation_rate=0.25)  # 25% max participation
    
    # Run backtest with intraday execution features
    backtester = aw.engine.vector.VectorBacktester()
    result = backtester.run(
        DailySignalIntradayExec,
        data={"TQQQ": frame},
        capital=100000.0,
        execution_price_model=price_model,
        trade_at_next_open=True,  # Execute at next day's open
        volume_limit=volume_model,  # Limit participation per bar
    )
    
    # Print results
    print(f"Final equity: ${result.final_equity:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe():.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Number of trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

