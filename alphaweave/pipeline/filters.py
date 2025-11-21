"""Filters for universe screening."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from alphaweave.core.frame import Frame
from alphaweave.pipeline.factors import Factor
from alphaweave.pipeline.expressions import FactorExpression


class Filter(ABC):
    """Base class for filters that produce boolean masks per symbol."""

    def __init__(self, name: Optional[str] = None):
        """
        Initialize filter.

        Args:
            name: Optional name for the filter
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Compute filter mask.

        Args:
            data: Dictionary mapping symbol names to Frame objects
            factors: Optional dictionary of precomputed factors

        Returns:
            DataFrame with boolean values: index=datetime, columns=symbols
        """
        raise NotImplementedError

    def __and__(self, other: "Filter") -> "And":
        """Combine filters with AND."""
        return And(self, other)

    def __or__(self, other: "Filter") -> "Or":
        """Combine filters with OR."""
        return Or(self, other)

    def __invert__(self) -> "Not":
        """Negate filter."""
        return Not(self)


def _align_dataframes(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align multiple DataFrames to a common datetime index."""
    if not frames:
        return pd.DataFrame()

    all_indices = set()
    for df in frames.values():
        all_indices.update(df.index)

    master_index = pd.DatetimeIndex(sorted(all_indices))
    aligned = {}
    for symbol, df in frames.items():
        aligned[symbol] = df.reindex(master_index)

    return pd.DataFrame(aligned)


class TopN(Filter):
    """Filter to select top N symbols by a factor."""

    def __init__(self, factor: Union[str, Factor, FactorExpression], n: int, ascending: bool = False):
        """
        Initialize TopN filter.

        Args:
            factor: Factor name (string), Factor instance, or FactorExpression
            n: Number of top symbols to select
            ascending: If True, select bottom N (ascending order)
        """
        super().__init__(f"TopN({n})")
        self.factor = factor
        self.n = n
        self.ascending = ascending

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute TopN filter mask."""
        if isinstance(self.factor, str):
            if factors is None or self.factor not in factors:
                return pd.DataFrame()
            factor_df = factors[self.factor]
        elif isinstance(self.factor, (Factor, FactorExpression)):
            factor_df = self.factor.compute(data)
        else:
            return pd.DataFrame()

        # For each row (datetime), select top N symbols
        mask = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)
        for idx in factor_df.index:
            row = factor_df.loc[idx].dropna()
            if len(row) == 0:
                continue
            top_n = row.nlargest(self.n) if not self.ascending else row.nsmallest(self.n)
            mask.loc[idx, top_n.index] = True

        return mask


class BottomN(Filter):
    """Filter to select bottom N symbols by a factor (alias for TopN with ascending=True)."""

    def __init__(self, factor: Union[str, Factor, FactorExpression], n: int):
        """
        Initialize BottomN filter.

        Args:
            factor: Factor name (string), Factor instance, or FactorExpression
            n: Number of bottom symbols to select
        """
        super().__init__(f"BottomN({n})")
        self.factor = factor
        self.n = n

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute BottomN filter mask."""
        return TopN(self.factor, self.n, ascending=True).compute(data, factors)


class PercentileFilter(Filter):
    """Filter symbols by percentile of a factor."""

    def __init__(
        self,
        factor: Union[str, Factor, FactorExpression],
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        ascending: bool = False,
    ):
        """
        Initialize percentile filter.

        Args:
            factor: Factor name (string), Factor instance, or FactorExpression
            top: Top percentile threshold (0-100)
            bottom: Bottom percentile threshold (0-100)
            ascending: If True, select low values (for volatility, etc.)
        """
        super().__init__("PercentileFilter")
        self.factor = factor
        self.top = top
        self.bottom = bottom
        self.ascending = ascending

        if top is None and bottom is None:
            raise ValueError("Either 'top' or 'bottom' must be specified")

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute percentile filter mask."""
        if isinstance(self.factor, str):
            if factors is None or self.factor not in factors:
                return pd.DataFrame()
            factor_df = factors[self.factor]
        elif isinstance(self.factor, (Factor, FactorExpression)):
            factor_df = self.factor.compute(data)
        else:
            return pd.DataFrame()

        mask = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)
        for idx in factor_df.index:
            row = factor_df.loc[idx].dropna()
            if len(row) == 0:
                continue

            if self.top is not None:
                threshold = row.quantile(1 - self.top / 100.0)
                if self.ascending:
                    mask.loc[idx, row[row <= threshold].index] = True
                else:
                    mask.loc[idx, row[row >= threshold].index] = True
            elif self.bottom is not None:
                threshold = row.quantile(self.bottom / 100.0)
                if self.ascending:
                    mask.loc[idx, row[row <= threshold].index] = True
                else:
                    mask.loc[idx, row[row >= threshold].index] = True

        return mask


class LiquidityFilter(Filter):
    """Filter symbols by liquidity (dollar volume)."""

    def __init__(self, top: Optional[int] = None, min_dollar_volume: Optional[float] = None):
        """
        Initialize liquidity filter.

        Args:
            top: Select top N symbols by dollar volume
            min_dollar_volume: Minimum dollar volume threshold
        """
        super().__init__("LiquidityFilter")
        self.top = top
        self.min_dollar_volume = min_dollar_volume

        if top is None and min_dollar_volume is None:
            raise ValueError("Either 'top' or 'min_dollar_volume' must be specified")

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute liquidity filter mask."""
        from alphaweave.pipeline.factors import DollarVolumeFactor

        dollar_vol_factor = DollarVolumeFactor()
        dollar_vol_df = dollar_vol_factor.compute(data)

        mask = pd.DataFrame(False, index=dollar_vol_df.index, columns=dollar_vol_df.columns)

        for idx in dollar_vol_df.index:
            row = dollar_vol_df.loc[idx].dropna()
            if len(row) == 0:
                continue

            if self.top is not None:
                top_symbols = row.nlargest(self.top).index
                mask.loc[idx, top_symbols] = True
            elif self.min_dollar_volume is not None:
                mask.loc[idx, row[row >= self.min_dollar_volume].index] = True

        return mask


class VolatilityFilter(Filter):
    """Filter symbols by volatility."""

    def __init__(self, percentile: Optional[float] = None, max_vol: Optional[float] = None, ascending: bool = True):
        """
        Initialize volatility filter.

        Args:
            percentile: Percentile threshold (0-100). If ascending=True, select bottom percentile.
            max_vol: Maximum volatility threshold
            ascending: If True, select low volatility (default: True)
        """
        super().__init__("VolatilityFilter")
        self.percentile = percentile
        self.max_vol = max_vol
        self.ascending = ascending

        if percentile is None and max_vol is None:
            raise ValueError("Either 'percentile' or 'max_vol' must be specified")

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute volatility filter mask."""
        from alphaweave.pipeline.factors import VolatilityFactor

        vol_factor = VolatilityFactor()
        vol_df = vol_factor.compute(data)

        mask = pd.DataFrame(False, index=vol_df.index, columns=vol_df.columns)

        for idx in vol_df.index:
            row = vol_df.loc[idx].dropna()
            if len(row) == 0:
                continue

            if self.percentile is not None:
                threshold = row.quantile(self.percentile / 100.0)
                if self.ascending:
                    mask.loc[idx, row[row <= threshold].index] = True
                else:
                    mask.loc[idx, row[row >= threshold].index] = True
            elif self.max_vol is not None:
                mask.loc[idx, row[row <= self.max_vol].index] = True

        return mask


class And(Filter):
    """Combine filters with AND logic."""

    def __init__(self, *filters: Filter):
        """
        Initialize AND filter.

        Args:
            *filters: Filters to combine
        """
        super().__init__("And")
        self.filters = filters

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute AND filter mask."""
        if not self.filters:
            return pd.DataFrame()

        result = self.filters[0].compute(data, factors)
        for f in self.filters[1:]:
            other = f.compute(data, factors)
            # Align and combine
            aligned = result.align(other, join="inner")
            result = aligned[0] & aligned[1]

        return result


class Or(Filter):
    """Combine filters with OR logic."""

    def __init__(self, *filters: Filter):
        """
        Initialize OR filter.

        Args:
            *filters: Filters to combine
        """
        super().__init__("Or")
        self.filters = filters

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute OR filter mask."""
        if not self.filters:
            return pd.DataFrame()

        result = self.filters[0].compute(data, factors)
        for f in self.filters[1:]:
            other = f.compute(data, factors)
            # Align and combine
            aligned = result.align(other, join="inner")
            result = aligned[0] | aligned[1]

        return result


class Not(Filter):
    """Negate a filter."""

    def __init__(self, filter: Filter):
        """
        Initialize NOT filter.

        Args:
            filter: Filter to negate
        """
        super().__init__("Not")
        self.filter = filter

    def compute(self, data: Dict[str, Frame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Compute NOT filter mask."""
        result = self.filter.compute(data, factors)
        return ~result

