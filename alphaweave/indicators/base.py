"""Base indicator class for alphaweave."""

from typing import Optional, Any, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame


class Indicator(ABC):
    """Base class for technical indicators with lazy per-bar evaluation."""

    def __init__(
        self,
        source: Union[Frame, pd.Series, np.ndarray, list],
        column: str = "close",
    ):
        """
        Initialize indicator.

        Args:
            source: Data source - Frame, pandas Series, numpy array, or list
            column: Column name to use (only for Frame sources, default: "close")
        """
        self.source = source
        self.column = column
        self._values: Optional[pd.Series] = None
        self._series: Optional[pd.Series] = None

    def _get_data(self) -> pd.Series:
        """
        Get pandas Series from source.
        
        Returns:
            pandas Series with the data to compute on
        """
        if self._series is None:
            if isinstance(self.source, Frame):
                # Frame: extract the specified column
                pdf = self.source.to_pandas()
                if self.column not in pdf.columns:
                    raise ValueError(f"Column '{self.column}' not found in Frame")
                self._series = pdf[self.column]
            elif isinstance(self.source, pd.Series):
                # Already a Series: use directly
                self._series = self.source
            elif isinstance(self.source, (np.ndarray, list)):
                # Array or list: convert to Series
                self._series = pd.Series(self.source)
            else:
                raise TypeError(
                    f"Unsupported source type: {type(self.source)}. "
                    "Expected Frame, pandas.Series, numpy.ndarray, or list"
                )
        return self._series

    def __getitem__(self, index: Any) -> float:
        """
        Get indicator value at a specific index (lazy evaluation).

        Args:
            index: Bar index (integer or pandas.Timestamp)

        Returns:
            Indicator value at the given index
        """
        if self._values is None:
            self._compute_all()
        
        if isinstance(index, int):
            # Integer index - use iloc
            if index < len(self._values):
                return float(self._values.iloc[index])
            return float("nan")
        else:
            # Timestamp index - use loc
            if index in self._values.index:
                return float(self._values.loc[index])
            return float("nan")

    def _compute_all(self) -> None:
        """Compute all indicator values at once (lazy evaluation)."""
        series = self._get_data()
        self._values = self.compute(series)

    @abstractmethod
    def compute(self, series: pd.Series) -> pd.Series:
        """
        Compute indicator values for the entire series.

        Args:
            series: Input price series

        Returns:
            Series with indicator values (same index as input)
        """
        raise NotImplementedError

    def values(self) -> pd.Series:
        """
        Get all computed indicator values.

        Returns:
            Series with all indicator values
        """
        if self._values is None:
            self._compute_all()
        return self._values.copy()

