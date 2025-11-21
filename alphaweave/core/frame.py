"""Canonical Frame abstraction wrapping a pandas or polars DataFrame."""

from typing import Any
import pandas as pd
import polars as pl
from alphaweave.utils.time import ensure_datetime_index


class Frame:
    """Canonical Frame abstraction wrapping a pandas or polars DataFrame."""

    def __init__(self, backend_obj: Any):
        """
        Initialize Frame with a pandas or polars DataFrame.

        Args:
            backend_obj: pandas DataFrame or polars DataFrame
        """
        if isinstance(backend_obj, pd.DataFrame):
            self.backend = "pandas"
            self._data = backend_obj.copy()
        elif isinstance(backend_obj, pl.DataFrame):
            self.backend = "polars"
            self._data = backend_obj
        else:
            raise TypeError(
                f"backend_obj must be pandas.DataFrame or polars.DataFrame, got {type(backend_obj)}"
            )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Frame":
        """
        Create Frame from pandas DataFrame and validate/normalize columns.

        Args:
            df: pandas DataFrame

        Returns:
            Frame instance
        """
        df = df.copy()
        df = cls._normalize_columns(df)
        df = ensure_datetime_index(df)
        frame = cls(df)
        frame.validate()
        return frame

    @classmethod
    def from_polars(cls, pl_df: pl.DataFrame) -> "Frame":
        """
        Create Frame from polars DataFrame and validate/normalize columns.

        Args:
            pl_df: polars DataFrame

        Returns:
            Frame instance
        """
        # Convert to pandas for normalization
        pdf = pl_df.to_pandas()
        pdf = cls._normalize_columns(pdf)
        pdf = ensure_datetime_index(pdf)
        # Reset index to column before converting back to polars (polars doesn't preserve index)
        pdf = pdf.reset_index()
        # Ensure datetime column is named "datetime"
        if pdf.index.name == "datetime" or (len(pdf.columns) > 0 and pdf.columns[0] == "datetime"):
            # Already has datetime column
            pass
        elif isinstance(pdf.index, pd.DatetimeIndex):
            # This shouldn't happen after reset_index, but handle it
            if pdf.index.name:
                pdf = pdf.rename(columns={pdf.index.name: "datetime"})
            else:
                pdf = pdf.rename(columns={"index": "datetime"})
        # Find datetime column and ensure it's named "datetime"
        datetime_cols = ["datetime", "dt", "timestamp"]
        for col in datetime_cols:
            if col in pdf.columns:
                if col != "datetime":
                    pdf = pdf.rename(columns={col: "datetime"})
                break
        # Convert back to polars
        pl_df_normalized = pl.from_pandas(pdf)
        frame = cls(pl_df_normalized)
        frame.validate()
        return frame

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert Frame to pandas DataFrame.

        Returns:
            pandas DataFrame
        """
        if self.backend == "pandas":
            return self._data.copy()
        else:
            pdf = self._data.to_pandas()
            # If the polars DataFrame has a datetime column, set it as index
            if "datetime" in pdf.columns:
                pdf["datetime"] = pd.to_datetime(pdf["datetime"])
                pdf = pdf.set_index("datetime")
            return pdf

    def to_polars(self) -> pl.DataFrame:
        """
        Convert Frame to polars DataFrame.

        Returns:
            polars DataFrame
        """
        if self.backend == "polars":
            return self._data.clone()
        else:
            # Reset index to column so polars can preserve it
            pdf = self._data.reset_index()
            # Ensure datetime column is named "datetime"
            if pdf.index.name == "datetime" or (len(pdf.columns) > 0 and pdf.columns[0] == "datetime"):
                # Already has datetime column
                pass
            elif isinstance(self._data.index, pd.DatetimeIndex):
                # Index is datetime, reset_index will create a column with index name
                if self._data.index.name:
                    pdf = pdf.rename(columns={self._data.index.name: "datetime"})
                else:
                    pdf = pdf.rename(columns={"index": "datetime"})
            return pl.from_pandas(pdf)

    def validate(self) -> None:
        """
        Raise ValueError if datetime column/index missing or no required columns.

        Raises:
            ValueError: If validation fails
        """
        pdf = self.to_pandas()

        # Check for datetime index
        if not isinstance(pdf.index, pd.DatetimeIndex):
            raise ValueError(
                f"Frame must have a datetime index. Got index type: {type(pdf.index)}"
            )

        # Check for required OHLC columns (at minimum)
        required_cols = {"open", "high", "low", "close"}
        available_cols = {col.lower() for col in pdf.columns}
        missing = required_cols - available_cols
        if missing:
            raise ValueError(
                f"Frame missing required columns: {missing}. "
                f"Available columns: {list(pdf.columns)}"
            )

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to canonical form (lowercase, standard names).

        Args:
            df: pandas DataFrame

        Returns:
            DataFrame with normalized columns
        """
        df = df.copy()

        # Column name mapping: common variations to canonical names
        column_mapping = {
            # Datetime variations
            "timestamp": "datetime",
            "dt": "datetime",
            "date": "datetime",
            "time": "datetime",
            # OHLCV variations
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vol": "volume",
            # Case variations (handle after lowercasing)
        }

        # First, lowercase all column names
        df.columns = [col.lower() for col in df.columns]

        # Apply mapping
        df = df.rename(columns=column_mapping)

        return df

