"""Factor expressions and transformations."""

from typing import Union, Optional
import pandas as pd
import numpy as np
from alphaweave.pipeline.factors import Factor


class FactorExpression:
    """Wrapper for factor expressions with operator overloads and transformations."""

    def __init__(self, factor: Union[Factor, "FactorExpression"]):
        """
        Initialize factor expression.

        Args:
            factor: Factor or FactorExpression to wrap
        """
        self._factor = factor
        self._transformations = []

    def _get_factor(self) -> Factor:
        """Get the underlying Factor object."""
        if isinstance(self._factor, FactorExpression):
            return self._factor._get_factor()
        return self._factor

    def compute(self, data: dict[str, "Frame"]) -> pd.DataFrame:
        """
        Compute the factor expression.

        Args:
            data: Dictionary mapping symbol names to Frame objects

        Returns:
            DataFrame with computed values
        """
        from alphaweave.core.frame import Frame

        # Compute base factor
        if isinstance(self._factor, FactorExpression):
            result = self._factor.compute(data)
        else:
            result = self._factor.compute(data)

        # Apply transformations in order
        for transform in self._transformations:
            result = transform(result)

        return result

    def zscore(self, window: Optional[int] = None) -> "FactorExpression":
        """
        Apply z-score normalization.

        Args:
            window: If provided, compute rolling z-score over time.
                   If None, compute cross-sectional z-score per bar.

        Returns:
            New FactorExpression with z-score transformation
        """
        expr = FactorExpression(self)
        if window is None:
            # Cross-sectional z-score per bar
            expr._transformations.append(lambda df: df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0))
        else:
            # Rolling z-score over time
            def rolling_zscore(df):
                result = df.copy()
                for col in df.columns:
                    result[col] = (df[col] - df[col].rolling(window=window, min_periods=1).mean()) / df[col].rolling(window=window, min_periods=1).std()
                return result
            expr._transformations.append(rolling_zscore)
        return expr

    def rank(self, ascending: bool = False) -> "FactorExpression":
        """
        Apply ranking transformation.

        Args:
            ascending: If True, rank ascending (lowest=1). If False, rank descending (highest=1).

        Returns:
            New FactorExpression with ranking transformation
        """
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: df.rank(axis=1, ascending=ascending, method="average"))
        return expr

    def percentile(self) -> "FactorExpression":
        """
        Apply percentile ranking (0-100).

        Returns:
            New FactorExpression with percentile transformation
        """
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: df.rank(axis=1, pct=True, method="average") * 100)
        return expr

    def mean(self, window: int) -> "FactorExpression":
        """
        Apply rolling mean.

        Args:
            window: Rolling window size

        Returns:
            New FactorExpression with rolling mean
        """
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: df.rolling(window=window, min_periods=1).mean())
        return expr

    def std(self, window: int) -> "FactorExpression":
        """
        Apply rolling standard deviation.

        Args:
            window: Rolling window size

        Returns:
            New FactorExpression with rolling std
        """
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: df.rolling(window=window, min_periods=1).std())
        return expr

    def __add__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Add two factors or add a constant."""
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Union[float, int]) -> "FactorExpression":
        """Right add (constant + factor)."""
        return self.__add__(other)

    def __sub__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Subtract two factors or subtract a constant."""
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Union[float, int]) -> "FactorExpression":
        """Right subtract (constant - factor)."""
        return FactorExpression(self).__neg__().__add__(other)

    def __mul__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Multiply two factors or multiply by a constant."""
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Union[float, int]) -> "FactorExpression":
        """Right multiply (constant * factor)."""
        return self.__mul__(other)

    def __truediv__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Divide two factors or divide by a constant."""
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Union[float, int]) -> "FactorExpression":
        """Right divide (constant / factor)."""
        # This is trickier - we need to compute factor first, then divide constant by it
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: other / df)
        return expr

    def __neg__(self) -> "FactorExpression":
        """Negate factor."""
        expr = FactorExpression(self)
        expr._transformations.append(lambda df: -df)
        return expr

    def __gt__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Compare factor > other."""
        return self._binary_op(other, lambda a, b: (a > b).astype(float))

    def __lt__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Compare factor < other."""
        return self._binary_op(other, lambda a, b: (a < b).astype(float))

    def __ge__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Compare factor >= other."""
        return self._binary_op(other, lambda a, b: (a >= b).astype(float))

    def __le__(self, other: Union[Factor, "FactorExpression", float, int]) -> "FactorExpression":
        """Compare factor <= other."""
        return self._binary_op(other, lambda a, b: (a <= b).astype(float))

    def _binary_op(
        self,
        other: Union[Factor, "FactorExpression", float, int],
        op,
    ) -> "FactorExpression":
        """Apply binary operation between two factors or factor and constant."""
        if isinstance(other, (float, int)):
            # Constant operation
            expr = FactorExpression(self)
            expr._transformations.append(lambda df: op(df, other))
            return expr
        else:
            # Factor operation - need to compute both and combine
            # Store the other factor/expression for later computation
            expr = FactorExpression(self)
            other_expr = FactorExpression(other) if isinstance(other, Factor) else other
            
            def combine(df1):
                # Compute other factor with same data (we'll need to pass data through)
                # For now, we'll compute it lazily - this is a limitation
                # In practice, the pipeline engine will handle this properly
                if isinstance(other, FactorExpression):
                    # Need to compute other factor - but we don't have data here
                    # This is a design limitation - we'll handle it in the pipeline engine
                    raise NotImplementedError("Factor-to-factor operations need to be handled by Pipeline engine")
                else:
                    # This shouldn't happen in practice
                    raise NotImplementedError("Factor-to-factor operations need to be handled by Pipeline engine")
            
            # For now, mark this as needing special handling
            # The pipeline engine will handle factor-to-factor operations
            expr._other_factor = other_expr
            expr._binary_op_func = op
            return expr

