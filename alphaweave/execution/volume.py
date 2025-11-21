"""Volume-based execution limits."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VolumeLimitModel:
    """
    Volume limit model for execution constraints.
    
    For intraday execution, max_participation_rate limits participation
    per bar to avoid market impact.
    """
    max_pct_volume: float = 1.0
    max_participation_rate: Optional[float] = None  # For intraday: max % of bar volume

    def clamp_size(self, desired_size: float, bar_volume: float) -> float:
        """
        Clamp order size based on volume limits.
        
        Args:
            desired_size: Desired order size
            bar_volume: Bar volume
            
        Returns:
            Clamped order size
        """
        if bar_volume is None or bar_volume <= 0:
            return 0.0
        
        # Apply participation rate limit for intraday
        if self.max_participation_rate is not None and self.max_participation_rate > 0:
            max_shares = self.max_participation_rate * bar_volume
        else:
            # Fall back to percentage of volume
            max_shares = self.max_pct_volume * bar_volume if self.max_pct_volume > 0 else 0.0
        
        if desired_size > 0:
            return min(desired_size, max_shares)
        else:
            return max(desired_size, -max_shares)
