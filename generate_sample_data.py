"""
generate_sample_data.py
Generates two synthetic energy-consumption sample datasets:
  1. data/sample_energy.csv       — 2 years of daily data
  2. data/sample_energy_hourly.csv — 60 days of hourly data

Run once before starting the Streamlit app:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)


def make_daily_dataset(
    start: str = "2023-01-01",
    days: int = 730,  # 2 years
) -> pd.DataFrame:
    """
    Generate a synthetic daily energy consumption dataset.

    Patterns included:
      - Long-term upward trend (slow growth)
      - Yearly seasonality (higher in winter months)
      - Weekly seasonality (weekdays > weekends)
      - Random Gaussian noise

    Returns:
        DataFrame with columns [timestamp, energy_consumption].
    """
    dates = pd.date_range(start=start, periods=days, freq="D")

    t = np.arange(days)

    # Long-term trend: slight upward drift
    trend = 100 + 0.02 * t

    # Yearly seasonality: peaks in Jan/Dec (winter heating)
    yearly = 20 * np.cos(2 * np.pi * t / 365.25 + np.pi)

    # Weekly seasonality: 10% higher on weekdays vs weekends
    day_of_week = dates.dayofweek  # 0=Mon, 6=Sun
    weekly = np.where(day_of_week < 5, 8.0, -8.0)  # weekday boost

    # Gaussian noise
    noise = RNG.normal(loc=0, scale=5, size=days)

    consumption = trend + yearly + weekly + noise
    # Clip to realistic non-negative values
    consumption = np.clip(consumption, 50, 250)

    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "energy_consumption": consumption.round(2),
    })
    return df


def make_hourly_dataset(
    start: str = "2024-01-01",
    days: int = 60,
) -> pd.DataFrame:
    """
    Generate a synthetic hourly energy consumption dataset.

    Patterns included:
      - Daily cycle (peaks in morning and evening)
      - Weekly seasonality (lower on weekends)
      - Short-term trend
      - Random noise

    Returns:
        DataFrame with columns [timestamp, energy_consumption].
    """
    hours = days * 24
    dates = pd.date_range(start=start, periods=hours, freq="h")

    t = np.arange(hours)

    # Trend
    trend = 80 + 0.005 * t

    # Daily cycle: two peaks — one around 9 AM, one around 7 PM
    hour_of_day = dates.hour
    daily = (
        12 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        + 8 * np.sin(4 * np.pi * (hour_of_day - 8) / 24)
    )

    # Weekly seasonality: lower on weekends
    is_weekend = (dates.dayofweek >= 5).astype(float)
    weekly = -10 * is_weekend

    # Noise
    noise = RNG.normal(loc=0, scale=4, size=hours)

    consumption = trend + daily + weekly + noise
    consumption = np.clip(consumption, 20, 200)

    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "energy_consumption": consumption.round(2),
    })
    return df


if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    daily_df = make_daily_dataset()
    daily_path = data_dir / "sample_energy.csv"
    daily_df.to_csv(daily_path, index=False)
    print(f"[OK] Daily dataset saved  -> {daily_path}  ({len(daily_df)} rows)")

    hourly_df = make_hourly_dataset()
    hourly_path = data_dir / "sample_energy_hourly.csv"
    hourly_df.to_csv(hourly_path, index=False)
    print(f"[OK] Hourly dataset saved -> {hourly_path}  ({len(hourly_df)} rows)")
