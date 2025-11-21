"""Tests for portfolio constraints."""

from alphaweave.portfolio.constraints import PortfolioConstraints, WeightBounds


def test_weight_bounds():
    """Test WeightBounds dataclass."""
    bounds = WeightBounds(lower=0.0, upper=0.1)
    assert bounds.lower == 0.0
    assert bounds.upper == 0.1


def test_portfolio_constraints_get_bounds():
    """Test PortfolioConstraints.get_bounds()."""
    constraints = PortfolioConstraints(
        weight_bounds={
            "AAPL": WeightBounds(lower=0.0, upper=0.2),
            "MSFT": WeightBounds(lower=0.0, upper=0.3),
        },
        default_bounds=WeightBounds(lower=0.0, upper=0.1),
        long_only=True,
    )

    # Asset with specific bounds
    lower, upper = constraints.get_bounds("AAPL")
    assert lower == 0.0
    assert upper == 0.2

    # Asset with default bounds
    lower, upper = constraints.get_bounds("GOOG")
    assert lower == 0.0
    assert upper == 0.1

    # Asset with no bounds specified
    constraints_no_default = PortfolioConstraints(long_only=True)
    lower, upper = constraints_no_default.get_bounds("AAPL")
    assert lower == 0.0
    assert upper == 1.0


def test_portfolio_constraints_get_bounds_arrays():
    """Test PortfolioConstraints.get_bounds_arrays()."""
    constraints = PortfolioConstraints(
        default_bounds=WeightBounds(lower=0.0, upper=0.5),
        long_only=True,
    )

    assets = ["A", "B", "C"]
    lower_bounds, upper_bounds = constraints.get_bounds_arrays(assets)

    assert len(lower_bounds) == 3
    assert len(upper_bounds) == 3
    assert all(lb == 0.0 for lb in lower_bounds)
    assert all(ub == 0.5 for ub in upper_bounds)

