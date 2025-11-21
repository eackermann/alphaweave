"""Tests for intraday partial fills with max_participation_rate."""

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.engine.vector import VectorBacktester
from alphaweave.execution.volume import VolumeLimitModel
from alphaweave.strategy.base import Strategy


class LargeOrderStrategy(Strategy):
    """Strategy that places a large order."""
    
    def init(self):
        pass
    
    def next(self, i):
        if i == 0:
            # Place a very large order (more than bar volume)
            self.order_target_percent("TEST", 1.0)


def test_max_participation_rate():
    """Test that max_participation_rate limits fills per bar."""
    # Create sample data with small volume
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-01-01", periods=10, freq="1h"),  # Hourly bars
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [100.0] * 10,  # Small volume per bar
    })
    
    frame = Frame.from_pandas(df)
    
    # Use max_participation_rate of 25%
    volume_model = VolumeLimitModel(max_participation_rate=0.25)
    
    backtester = VectorBacktester()
    result = backtester.run(
        LargeOrderStrategy,
        data={"TEST": frame},
        capital=10000.0,
        volume_limit=volume_model,
    )
    
    # Should have trades
    assert len(result.trades) > 0
    
    # Each fill should be limited by participation rate
    # With 25% participation and 100 volume, max fill per bar is 25 shares
    for trade in result.trades:
        assert abs(trade.size) <= 25.0  # Should be limited by participation rate

