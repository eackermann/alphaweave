"""Pipeline engine for computing factors and applying filters."""

from typing import Dict, Optional, Union
import pandas as pd
from alphaweave.core.frame import Frame
from alphaweave.pipeline.factors import Factor
from alphaweave.pipeline.filters import Filter
from alphaweave.pipeline.expressions import FactorExpression


class Pipeline:
    """
    Pipeline for computing factors and applying filters.

    A Pipeline consists of:
    - factors: Dictionary of factor names to Factor instances
    - filters: Dictionary of filter names to Filter instances
    - screen: Optional final filter to apply (combination of filters)
    """

    def __init__(
        self,
        factors: Optional[Dict[str, Union[Factor, FactorExpression]]] = None,
        filters: Optional[Dict[str, Filter]] = None,
        screen: Optional[Union[Filter, str]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            factors: Dictionary mapping factor names to Factor instances or expressions
            filters: Dictionary mapping filter names to Filter instances
            screen: Optional final screen filter (can be a Filter instance or filter name string)
        """
        self.factors = factors or {}
        self.filters = filters or {}
        self.screen = screen

    def run(
        self,
        data: Dict[str, Frame],
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        cache: bool = True,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Run the pipeline.

        Args:
            data: Dictionary mapping symbol names to Frame objects
            start: Optional start timestamp to slice data
            end: Optional end timestamp to slice data
            cache: Whether to cache intermediate results (default: True)

        Returns:
            Dictionary with keys:
            - "factors": Dictionary of factor name -> DataFrame
            - "filters": Dictionary of filter name -> DataFrame (boolean masks)
            - "screen": Final screen DataFrame (boolean mask)
        """
        # Slice data if needed
        if start is not None or end is not None:
            data = self._slice_data(data, start, end)

        # Compute all factors
        factor_results = {}
        for name, factor in self.factors.items():
            if isinstance(factor, FactorExpression):
                factor_df = factor.compute(data)
            elif isinstance(factor, Factor):
                factor_df = factor.compute(data)
            else:
                raise TypeError(f"Factor must be Factor or FactorExpression, got {type(factor)}")
            factor_results[name] = factor_df

        # Compute all filters
        filter_results = {}
        for name, filter_obj in self.filters.items():
            filter_df = filter_obj.compute(data, factors=factor_results)
            filter_results[name] = filter_df

        # Compute final screen if provided
        screen_result = None
        if self.screen is not None:
            if isinstance(self.screen, str):
                # Screen is a filter name
                if self.screen in filter_results:
                    screen_result = filter_results[self.screen]
                elif self.screen in self.filters:
                    screen_result = self.filters[self.screen].compute(data, factors=factor_results)
                else:
                    raise ValueError(f"Screen filter '{self.screen}' not found in filters")
            elif isinstance(self.screen, Filter):
                # Screen is a Filter instance
                screen_result = self.screen.compute(data, factors=factor_results)
            else:
                raise TypeError(f"Screen must be Filter or str, got {type(self.screen)}")

        return {
            "factors": factor_results,
            "filters": filter_results,
            "screen": screen_result,
        }

    def _slice_data(self, data: Dict[str, Frame], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Dict[str, Frame]:
        """
        Slice data by timestamp range.

        Args:
            data: Dictionary of symbol -> Frame
            start: Optional start timestamp
            end: Optional end timestamp

        Returns:
            Sliced data dictionary
        """
        sliced = {}
        for symbol, frame in data.items():
            df = frame.to_pandas()
            if start is not None:
                df = df.loc[df.index >= start]
            if end is not None:
                df = df.loc[df.index <= end]
            # Convert back to Frame
            from alphaweave.core.frame import Frame
            sliced[symbol] = Frame.from_pandas(df)
        return sliced

