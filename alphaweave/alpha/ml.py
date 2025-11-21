"""Optional machine learning helpers for factor modeling."""

from typing import Optional, Callable
import pandas as pd
import numpy as np


def fit_factor_model(
    factor_data: pd.DataFrame,
    future_returns: pd.Series,
    model: str = "linear",
    **model_kwargs,
) -> tuple[object, dict]:
    """
    Fit a model to predict future returns from factor values.

    Args:
        factor_data: DataFrame with factor values [index = date×symbol, columns = factors]
        future_returns: Series of future returns (aligned with factor_data index)
        model: Model type ("linear", "ridge", "random_forest")
        **model_kwargs: Additional arguments for the model

    Returns:
        Tuple of (fitted_model, diagnostics_dict)

    Raises:
        ImportError: If required ML library is not installed
    """
    # Align data
    aligned = pd.DataFrame({"returns": future_returns}).join(factor_data, how="inner")
    aligned = aligned.dropna()

    if len(aligned) == 0:
        raise ValueError("No aligned data after joining factor_data and future_returns")

    X = aligned[factor_data.columns].values
    y = aligned["returns"].values

    if model == "linear":
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise ImportError(
                "scikit-learn is required for linear model. Install with: pip install scikit-learn"
            )

        reg = LinearRegression(**model_kwargs)
        reg.fit(X, y)
        y_pred = reg.predict(X)

        diagnostics = {
            "r_squared": float(reg.score(X, y)),
            "coefficients": dict(zip(factor_data.columns, reg.coef_)),
            "intercept": float(reg.intercept_),
            "n_samples": len(aligned),
        }

        return reg, diagnostics

    elif model == "ridge":
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ridge model. Install with: pip install scikit-learn"
            )

        alpha = model_kwargs.get("alpha", 1.0)
        reg = Ridge(alpha=alpha, **{k: v for k, v in model_kwargs.items() if k != "alpha"})
        reg.fit(X, y)
        y_pred = reg.predict(X)

        diagnostics = {
            "r_squared": float(reg.score(X, y)),
            "coefficients": dict(zip(factor_data.columns, reg.coef_)),
            "intercept": float(reg.intercept_),
            "alpha": float(reg.alpha),
            "n_samples": len(aligned),
        }

        return reg, diagnostics

    elif model == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required for random forest. Install with: pip install scikit-learn"
            )

        n_estimators = model_kwargs.get("n_estimators", 100)
        max_depth = model_kwargs.get("max_depth", None)
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=model_kwargs.get("random_state", None),
            **{k: v for k, v in model_kwargs.items() if k not in ["n_estimators", "max_depth", "random_state"]},
        )
        rf.fit(X, y)
        y_pred = rf.predict(X)

        feature_importance = dict(zip(factor_data.columns, rf.feature_importances_))

        diagnostics = {
            "r_squared": float(rf.score(X, y)),
            "feature_importance": feature_importance,
            "n_samples": len(aligned),
            "n_estimators": n_estimators,
        }

        return rf, diagnostics

    else:
        raise ValueError(f"Unknown model type: {model}. Use 'linear', 'ridge', or 'random_forest'")


def model_to_scoring_function(
    model: object,
    factor_names: list[str],
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Convert a fitted model into a scoring function for use in strategies.

    Args:
        model: Fitted model (from fit_factor_model)
        factor_names: List of factor names (must match model input)

    Returns:
        Function that takes factor DataFrame and returns predicted returns Series
    """
    def scoring_function(factor_df: pd.DataFrame) -> pd.Series:
        """
        Score assets based on factor values.

        Args:
            factor_df: DataFrame with factor values [index = symbols, columns = factors]

        Returns:
            Series of predicted returns [index = symbols]
        """
        # Ensure columns match
        available_factors = [f for f in factor_names if f in factor_df.columns]
        if not available_factors:
            return pd.Series(0.0, index=factor_df.index)

        # Prepare input
        X = factor_df[available_factors].values

        # Predict
        predictions = model.predict(X)

        return pd.Series(predictions, index=factor_df.index)

    return scoring_function

