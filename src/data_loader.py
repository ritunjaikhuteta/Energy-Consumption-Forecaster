"""
src/data_loader.py
Handles CSV loading, column detection, and initial validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_csv(file_path_or_buffer) -> pd.DataFrame:
    """
    Load a CSV file from a file path or an uploaded file buffer.

    Args:
        file_path_or_buffer: A file path (str) or a file-like buffer (e.g., from st.file_uploader).

    Returns:
        A raw pandas DataFrame.

    Raises:
        ValueError: If the file cannot be read or is empty.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty. Please upload a file with data.")

    if len(df.columns) < 2:
        raise ValueError(
            "The CSV file must have at least two columns: a date/time column and a numeric energy column."
        )

    return df


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to automatically detect the datetime column from the DataFrame.

    Looks for columns that can be parsed as datetime objects.

    Args:
        df: Raw DataFrame.

    Returns:
        Column name string if found, else None.
    """
    for col in df.columns:
        # Check for common datetime-related names first
        if any(kw in col.lower() for kw in ["date", "time", "timestamp", "datetime", "ds"]):
            return col

    # Fall back: try to parse each column as datetime
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.8:
                return col
        except Exception:
            continue

    return None


def detect_value_column(df: pd.DataFrame, datetime_col: str) -> Optional[str]:
    """
    Try to automatically detect the numeric energy consumption column.

    Args:
        df: Raw DataFrame.
        datetime_col: The already-detected datetime column name (excluded from search).

    Returns:
        Column name string if found, else None.
    """
    for col in df.columns:
        if col == datetime_col:
            continue
        # Check for common energy-related names
        if any(kw in col.lower() for kw in ["energy", "consumption", "kwh", "power", "load", "usage", "value", "y"]):
            return col

    # Fall back: pick the first numeric column that is not the datetime column
    for col in df.columns:
        if col == datetime_col:
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > len(df) * 0.8:
                return col
        except Exception:
            continue

    return None


def prepare_prophet_dataframe(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
) -> Tuple[pd.DataFrame, list]:
    """
    Clean and convert the raw DataFrame into the Prophet-required format.

    Prophet requires exactly two columns: 'ds' (datetime) and 'y' (numeric target).

    Processing steps:
        1. Select the two relevant columns.
        2. Rename them to 'ds' and 'y'.
        3. Parse 'ds' as datetime.
        4. Parse 'y' as float.
        5. Drop rows with NaT or NaN.
        6. Remove duplicate timestamps (keep first).
        7. Sort chronologically.

    Args:
        df: Raw DataFrame.
        datetime_col: Name of the date/time column.
        value_col: Name of the energy consumption column.

    Returns:
        Tuple of (cleaned Prophet DataFrame, list of warning messages).

    Raises:
        ValueError: If the resulting DataFrame has too few valid rows.
    """
    warnings: list = []

    prophet_df = df[[datetime_col, value_col]].copy()
    prophet_df.columns = ["ds", "y"]

    # Parse datetime
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
    invalid_dates = prophet_df["ds"].isna().sum()
    if invalid_dates > 0:
        warnings.append(f"Removed {invalid_dates} row(s) with invalid/unparseable date values.")

    # Parse numeric
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
    invalid_values = prophet_df["y"].isna().sum()
    if invalid_values > 0:
        warnings.append(f"Removed {invalid_values} row(s) with non-numeric energy values.")

    # Drop rows with NaT or NaN
    prophet_df.dropna(subset=["ds", "y"], inplace=True)

    # Remove duplicates
    dup_count = prophet_df.duplicated(subset=["ds"]).sum()
    if dup_count > 0:
        warnings.append(f"Removed {dup_count} duplicate timestamp(s) (kept first occurrence).")
    prophet_df.drop_duplicates(subset=["ds"], keep="first", inplace=True)

    # Sort chronologically
    prophet_df.sort_values("ds", inplace=True)
    prophet_df.reset_index(drop=True, inplace=True)

    if len(prophet_df) < 10:
        raise ValueError(
            f"After cleaning, only {len(prophet_df)} valid rows remain. "
            "Prophet requires at least 10 observations to train. "
            "Please provide a larger dataset."
        )

    return prophet_df, warnings


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a cleaned Prophet DataFrame (ds + y).

    Args:
        df: Cleaned DataFrame with 'ds' and 'y' columns.

    Returns:
        Dictionary with summary stats:
            - num_records
            - start_date
            - end_date
            - avg_consumption
            - min_consumption
            - max_consumption
            - date_range_days
            - estimated_frequency
    """
    num_records = len(df)
    start_date = df["ds"].min()
    end_date = df["ds"].max()
    avg_consumption = df["y"].mean()
    min_consumption = df["y"].min()
    max_consumption = df["y"].max()
    date_range_days = (end_date - start_date).days

    # Estimate frequency
    if num_records > 1:
        median_diff = df["ds"].diff().dropna().median()
        hours = median_diff.total_seconds() / 3600
        if hours <= 1.5:
            freq = "Hourly"
        elif hours <= 25:
            freq = "Daily"
        elif hours <= 170:
            freq = "Weekly"
        else:
            freq = "Monthly or less frequent"
    else:
        freq = "Unknown"

    return {
        "num_records": num_records,
        "start_date": start_date,
        "end_date": end_date,
        "avg_consumption": avg_consumption,
        "min_consumption": min_consumption,
        "max_consumption": max_consumption,
        "date_range_days": date_range_days,
        "estimated_frequency": freq,
    }
