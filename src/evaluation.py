"""
src/evaluation.py
Computes time-series model evaluation metrics and prepares test-period predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def predict_test_period(
    model: Prophet,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Prophet predictions for the exact timestamps in the test DataFrame.

    Creates a future DataFrame from the test dates and calls model.predict().

    Args:
        model: A fitted Prophet model.
        test_df: Test DataFrame with 'ds' and 'y' columns.

    Returns:
        Prophet forecast DataFrame for the test period.
    """
    future_test = test_df[["ds"]].copy()
    forecast_test = model.predict(future_test)
    return forecast_test


def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE between actual and predicted values.

    - MAE  (Mean Absolute Error): Average absolute difference.
    - RMSE (Root Mean Squared Error): Square root of average squared differences;
            penalises large errors more heavily.
    - MAPE (Mean Absolute Percentage Error): Average percentage error;
            useful for comparing across different scales.

    Args:
        actual: Series of ground-truth values.
        predicted: Series of model predictions.

    Returns:
        Dict with keys 'MAE', 'RMSE', 'MAPE'.
    """
    actual_arr = np.array(actual, dtype=float)
    predicted_arr = np.array(predicted, dtype=float)

    mae = mean_absolute_error(actual_arr, predicted_arr)
    rmse = np.sqrt(mean_squared_error(actual_arr, predicted_arr))

    # MAPE: avoid division by zero
    nonzero_mask = actual_arr != 0
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = np.mean(np.abs((actual_arr[nonzero_mask] - predicted_arr[nonzero_mask]) / actual_arr[nonzero_mask])) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def merge_actuals_and_predictions(
    test_df: pd.DataFrame,
    forecast_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the test DataFrame (actual values) with Prophet forecast output.

    The result contains: ds, y (actual), yhat, yhat_lower, yhat_upper.

    Args:
        test_df: Test DataFrame with 'ds' and 'y' columns.
        forecast_test: Prophet forecast DataFrame for the test period.

    Returns:
        Merged DataFrame with both actual and predicted columns.
    """
    merged = pd.merge(
        test_df[["ds", "y"]],
        forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )
    return merged
