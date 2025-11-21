"""Universe selection and ranking utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def top_n_by_score(
    scores: pd.Series,
    n: int,
    *,
    ascending: bool = False,
) -> Sequence[str]:
    """
    Select top-N assets by score.

    Args:
        scores: Series indexed by asset; higher is better by default
        n: Number of top assets to select
        ascending: If True, select bottom-N (lowest scores)

    Returns:
        List of asset identifiers
    """
    if n <= 0:
        return []
    if len(scores) == 0:
        return []

    sorted_scores = scores.sort_values(ascending=ascending)
    top_n = sorted_scores.head(n)
    return list(top_n.index)


def normalize_scores_to_weights(
    scores: pd.Series,
    *,
    long_only: bool = True,
) -> pd.Series:
    """
    Map arbitrary scores to portfolio weights.

    For long_only=True: set negative scores to 0, then normalize to sum=1.

    Args:
        scores: Series indexed by asset with arbitrary scores
        long_only: If True, set negative scores to 0 before normalizing

    Returns:
        Series of normalized weights (sums to 1.0)
    """
    if len(scores) == 0:
        return pd.Series(dtype=float)

    weights = scores.copy()

    if long_only:
        # Set negative scores to 0
        weights = weights.clip(lower=0.0)

    # Normalize to sum = 1
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        # If all scores are zero or negative, use equal weights
        weights = pd.Series([1.0 / len(scores)] * len(scores), index=scores.index)

    return weights

