"""Volume-based execution limits."""

from dataclasses import dataclass


@dataclass
class VolumeLimitModel:
    max_pct_volume: float = 1.0

    def clamp_size(self, desired_size: float, bar_volume: float) -> float:
        if bar_volume is None or bar_volume <= 0 or self.max_pct_volume <= 0:
            return 0.0
        max_shares = self.max_pct_volume * bar_volume
        if desired_size > 0:
            return min(desired_size, max_shares)
        else:
            return max(desired_size, -max_shares)
