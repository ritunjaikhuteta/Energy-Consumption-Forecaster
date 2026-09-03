"""
src/preprocessing.py
Handles data-frequency detection and train/test splitting for time-series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_frequency(df: pd.DataFrame) -> str:
    """
    Detect whether the dataset is hourly or daily based on the median time difference.

    Args:
        df: Prophet-formatted DataFrame with 'ds' column.

    Returns:
        'H' for hourly, 'D' for daily.
    """
    if len(df) < 2:
        return "D"

    median_diff = df["ds"].diff().dropna().median()
    hours = median_diff.total_seconds() / 3600

    if hours <= 1.5:
        return "h"
    return "D"


def train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame into train and test sets chronologically.

    The last `test_fraction` of observations form the test set.
    This ensures no data leakage from future to past.

    Args:
        df: Cleaned Prophet DataFrame (ds, y).
        test_fraction: Proportion of data to reserve for testing (default 0.2 = 20%).

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If there are too few rows to create a meaningful split.
    """
    n = len(df)
    if n < 20:
        raise ValueError(
            f"Dataset has only {n} rows. At least 20 rows are needed "
            "for a meaningful train/test split."
        )

    split_idx = int(n * (1 - test_fraction))
    # Ensure at least 10 training points
    split_idx = max(split_idx, 10)
    # Ensure at least 1 test point
    split_idx = min(split_idx, n - 1)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def check_seasonality_feasibility(df: pd.DataFrame, freq: str) -> dict:
    """
    Determine which seasonality components are feasible given the data length.

    Prophet requires enough historical data to fit each seasonality:
      - Daily seasonality:   >2 days of hourly data
      - Weekly seasonality:  >2 weeks of data
      - Yearly seasonality:  >2 full years of data (recommended by Prophet)

    Args:
        df: Cleaned Prophet DataFrame.
        freq: Detected frequency ('H' or 'D').

    Returns:
        Dict with boolean flags: daily, weekly, yearly.
    """
    date_range_days = (df["ds"].max() - df["ds"].min()).days
    n = len(df)

    feasibility = {
        "daily": False,
        "weekly": False,
        "yearly": False,
    }

    if freq == "h":
        # Daily seasonality: need at least 48 hourly observations (2 days)
        feasibility["daily"] = n >= 48
        # Weekly seasonality: need at least 2 weeks
        feasibility["weekly"] = date_range_days >= 14
    else:
        # For daily data, daily seasonality doesn't apply
        feasibility["daily"] = False
        # Weekly seasonality: need at least 2 weeks
        feasibility["weekly"] = date_range_days >= 14

    # Yearly seasonality: need at least 2 years
    feasibility["yearly"] = date_range_days >= 365 * 2

    return feasibility
