"""
src/forecasting.py
Handles Prophet model initialization, training, and forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple


def build_prophet_model(
    freq: str,
    seasonality_feasibility: dict,
    yearly_seasonality_override: bool = False,
) -> Prophet:
    """
    Build and configure a Prophet model based on the dataset frequency and feasibility.

    Prophet is configured differently for hourly vs. daily data:
      - Hourly data: daily + weekly seasonality enabled if feasible.
      - Daily data: weekly + yearly seasonality enabled if feasible.
    Yearly seasonality is only enabled when >=2 years of data are present,
    unless overridden by the user.

    Args:
        freq: Dataset frequency - 'H' (hourly) or 'D' (daily).
        seasonality_feasibility: Dict from preprocessing.check_seasonality_feasibility().
        yearly_seasonality_override: Allow the user to force yearly seasonality on.

    Returns:
        A configured (but not yet fitted) Prophet model.
    """
    enable_daily = seasonality_feasibility.get("daily", False) and freq == "h"
    enable_weekly = seasonality_feasibility.get("weekly", True)
    enable_yearly = seasonality_feasibility.get("yearly", False) or yearly_seasonality_override

    model = Prophet(
        daily_seasonality=enable_daily,
        weekly_seasonality=enable_weekly,
        yearly_seasonality=enable_yearly,
        interval_width=0.95,
    )

    return model


def fit_model(model: Prophet, train_df: pd.DataFrame) -> Prophet:
    """
    Fit a Prophet model to the training DataFrame.

    Args:
        model: A configured Prophet model instance.
        train_df: Training DataFrame with 'ds' and 'y' columns.

    Returns:
        The fitted Prophet model.

    Raises:
        RuntimeError: If fitting fails.
    """
    try:
        model.fit(train_df)
    except Exception as e:
        raise RuntimeError(f"Prophet model training failed: {e}")
    return model


def generate_forecast(
    model: Prophet,
    periods: int,
    freq: str,
    include_history: bool = True,
) -> pd.DataFrame:
    """
    Create a future dataframe and generate forecasts.

    Args:
        model: A fitted Prophet model.
        periods: Number of future periods to forecast.
        freq: Frequency string - 'h' for hourly, 'D' for daily.
        include_history: Whether to include historical dates in the future frame.

    Returns:
        Prophet forecast DataFrame containing ds, yhat, yhat_lower, yhat_upper, trend, etc.
    """
    future = model.make_future_dataframe(
        periods=periods,
        freq=freq,
        include_history=include_history,
    )

    forecast = model.predict(future)
    return forecast


def calculate_forecast_periods(forecast_days: int, freq: str) -> int:
    """
    Convert a number of forecast days to the correct number of Prophet periods.

    Args:
        forecast_days: Desired forecast horizon in days.
        freq: Frequency - 'h' or 'D'.

    Returns:
        Number of periods (int).
    """
    if freq == "h":
        return forecast_days * 24
    return forecast_days


def extract_future_forecast(
    forecast: pd.DataFrame,
    train_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract only the future predictions from the full Prophet forecast DataFrame.

    Filters rows where 'ds' is strictly after the last training date.

    Args:
        forecast: Full Prophet forecast DataFrame.
        train_end_date: The last date in the training dataset.

    Returns:
        A filtered DataFrame containing only future predictions.
    """
    future_mask = forecast["ds"] > train_end_date
    future_df = forecast.loc[future_mask, ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    future_df.reset_index(drop=True, inplace=True)
    return future_df
