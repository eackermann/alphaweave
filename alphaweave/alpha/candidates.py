"""Strategy candidate specifications for discovery."""

from dataclasses import dataclass
from typing import Callable, Mapping, Any, Type, Optional
from alphaweave.strategy.base import Strategy
from alphaweave.alpha.search_space import SearchSpace


StrategyFactory = Callable[[Mapping[str, Any]], Type[Strategy]]
"""
Factory function that takes parameter dictionary and returns Strategy class.

Example:
    def factory(params: dict) -> Type[Strategy]:
        fast = params["fast"]
        slow = params["slow"]
        
        class SMACross(Strategy):
            def init(self):
                self.fast = fast
                self.slow = slow
            def next(self, i):
                ...
        
        return SMACross
"""


@dataclass
class StrategyCandidateSpec:
    """
    Specification for a strategy candidate to be evaluated.

    Attributes:
        name: Name identifier for this candidate
        factory: Function that creates Strategy class from parameters
        search_space: Search space for parameters
        metadata: Optional metadata dictionary
    """

    name: str
    factory: StrategyFactory
    search_space: SearchSpace
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """Validate candidate spec."""
        if not self.name:
            raise ValueError("StrategyCandidateSpec must have a name")

        if not callable(self.factory):
            raise ValueError("factory must be callable")

