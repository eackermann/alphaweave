"""Risk constraints for portfolio execution."""

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_symbol_weight: float = 1.0
    max_gross_leverage: float = 1.0
