"""Base signal class for alphaweave."""

from typing import Any
from abc import ABC, abstractmethod


class Signal(ABC):
    """Base class for trading signals."""

    @abstractmethod
    def __call__(self, index: Any) -> bool:
        """
        Evaluate signal at a specific index.

        Args:
            index: Bar index (integer or pandas.Timestamp)

        Returns:
            True if signal is triggered, False otherwise
        """
        raise NotImplementedError

