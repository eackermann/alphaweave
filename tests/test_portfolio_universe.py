"""Tests for universe selection and ranking utilities."""

import pandas as pd

from alphaweave.portfolio.universe import normalize_scores_to_weights, top_n_by_score


def test_top_n_by_score():
    """Test top_n_by_score function."""
    scores = pd.Series({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.4, "E": 0.05})

    # Top 3 (highest scores)
    top_3 = top_n_by_score(scores, n=3, ascending=False)
    assert len(top_3) == 3
    assert "D" in top_3  # Highest
    assert "B" in top_3
    assert "C" in top_3

    # Bottom 2 (lowest scores)
    bottom_2 = top_n_by_score(scores, n=2, ascending=True)
    assert len(bottom_2) == 2
    assert "E" in bottom_2  # Lowest
    assert "A" in bottom_2


def test_normalize_scores_to_weights():
    """Test normalize_scores_to_weights function."""
    scores = pd.Series({"A": 0.1, "B": 0.3, "C": 0.2})

    weights = normalize_scores_to_weights(scores, long_only=True)

    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w >= 0 for w in weights.values)  # Long-only
    # Higher scores should get higher weights
    assert weights["B"] > weights["A"]


def test_normalize_scores_to_weights_with_negative():
    """Test normalization with negative scores."""
    scores = pd.Series({"A": 0.1, "B": -0.2, "C": 0.3})

    weights = normalize_scores_to_weights(scores, long_only=True)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert weights["B"] == 0.0  # Negative scores set to 0
    assert weights["C"] > weights["A"]  # Higher positive score gets higher weight


def test_normalize_scores_to_weights_all_negative():
    """Test normalization when all scores are negative."""
    scores = pd.Series({"A": -0.1, "B": -0.2, "C": -0.3})

    weights = normalize_scores_to_weights(scores, long_only=True)

    # Should fall back to equal weights
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(abs(w - 1.0 / 3) < 1e-6 for w in weights.values)

