"""
src/visualization.py
All chart-generation functions used by the Streamlit dashboard.
Uses Plotly for interactive charts and Matplotlib for Prophet component plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Historical Data Charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_historical(df: pd.DataFrame, rolling_window: Optional[int] = None) -> go.Figure:
    """
    Plot historical energy consumption as an interactive Plotly line chart.

    Optionally overlays a rolling average to smooth short-term fluctuations.

    Args:
        df: Cleaned Prophet DataFrame (ds, y).
        rolling_window: If provided, number of periods for rolling average.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    # Raw consumption line
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines",
        name="Energy Consumption",
        line=dict(color="#00B4D8", width=1.2),
        opacity=0.85,
    ))

    # Rolling average overlay
    if rolling_window and rolling_window > 1:
        rolling_avg = df["y"].rolling(window=rolling_window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df["ds"],
            y=rolling_avg,
            mode="lines",
            name=f"{rolling_window}-period Rolling Avg",
            line=dict(color="#FF6B6B", width=2.5, dash="solid"),
        ))

    fig.update_layout(
        title=dict(text="📊 Historical Energy Consumption", font=dict(size=20)),
        xaxis_title="Date / Time",
        yaxis_title="Energy Consumption",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


def plot_daily_pattern(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plot average energy consumption by day of week (for daily or hourly datasets).

    Args:
        df: Cleaned Prophet DataFrame (ds, y).

    Returns:
        A Plotly Figure object, or None if dataset is too small.
    """
    if len(df) < 14:
        return None

    temp = df.copy()
    temp["day_of_week"] = temp["ds"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_by_day = (
        temp.groupby("day_of_week")["y"]
        .mean()
        .reindex(day_order)
        .dropna()
        .reset_index()
    )
    avg_by_day.columns = ["Day", "Avg Consumption"]

    fig = px.bar(
        avg_by_day,
        x="Day",
        y="Avg Consumption",
        color="Avg Consumption",
        color_continuous_scale="Viridis",
        title="📅 Average Consumption by Day of Week",
        template="plotly_dark",
    )
    fig.update_layout(height=380, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_monthly_pattern(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plot average energy consumption by month of year.

    Args:
        df: Cleaned Prophet DataFrame (ds, y).

    Returns:
        A Plotly Figure object, or None if dataset spans less than 2 months.
    """
    if (df["ds"].max() - df["ds"].min()).days < 60:
        return None

    temp = df.copy()
    temp["month"] = temp["ds"].dt.month
    temp["month_name"] = temp["ds"].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_by_month = (
        temp.groupby(["month", "month_name"])["y"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    fig = px.line(
        avg_by_month,
        x="month_name",
        y="y",
        markers=True,
        title="📆 Average Consumption by Month",
        labels={"month_name": "Month", "y": "Avg Consumption"},
        template="plotly_dark",
        color_discrete_sequence=["#48CAE4"],
        category_orders={"month_name": month_order},
    )
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Forecast Charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_forecast_interactive(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    show_confidence: bool = True,
) -> go.Figure:
    """
    Interactive Plotly chart displaying historical data and future forecast.

    Clearly distinguishes historical points (blue) from predicted future points (orange).
    Optionally shows the 95% confidence interval ribbon.

    Args:
        df: Full cleaned Prophet DataFrame (all historical ds + y).
        forecast: Full Prophet forecast DataFrame.
        show_confidence: Whether to display yhat_lower / yhat_upper confidence bands.

    Returns:
        A Plotly Figure object.
    """
    train_end = df["ds"].max()
    future_forecast = forecast[forecast["ds"] > train_end]

    fig = go.Figure()

    # Historical actual values
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines",
        name="Historical",
        line=dict(color="#00B4D8", width=1.5),
    ))

    # Confidence interval ribbon (future only)
    if show_confidence and not future_forecast.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_forecast["ds"], future_forecast["ds"].iloc[::-1]]),
            y=pd.concat([future_forecast["yhat_upper"], future_forecast["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            showlegend=True,
        ))

    # Future predicted values
    if not future_forecast.empty:
        fig.add_trace(go.Scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#FF6B6B", width=2.5),
        ))

    # Vertical dashed line at forecast start
    fig.add_vline(
        x=train_end,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text="Forecast Start",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text="🔮 Energy Consumption Forecast", font=dict(size=20)),
        xaxis_title="Date / Time",
        yaxis_title="Energy Consumption",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=40, r=20, t=70, b=40),
    )

    return fig


def plot_forecast_matplotlib(model: Prophet, forecast: pd.DataFrame) -> plt.Figure:
    """
    Generate the standard Prophet Matplotlib forecast plot.

    This is the canonical Prophet output showing historical data, forecast line,
    and uncertainty intervals.

    Args:
        model: Fitted Prophet model.
        forecast: Full Prophet forecast DataFrame.

    Returns:
        A Matplotlib Figure object.
    """
    fig = model.plot(forecast)
    fig.set_size_inches(12, 5)
    ax = fig.axes[0]
    ax.set_title("Prophet Forecast", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Consumption")
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    return fig


def plot_components_matplotlib(model: Prophet, forecast: pd.DataFrame) -> plt.Figure:
    """
    Generate Prophet's Matplotlib component plots (trend, seasonality, etc.).

    Args:
        model: Fitted Prophet model.
        forecast: Full Prophet forecast DataFrame.

    Returns:
        A Matplotlib Figure object.
    """
    fig = model.plot_components(forecast)
    fig.set_size_inches(12, 8)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in fig.axes:
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(merged_df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly chart comparing actual vs predicted energy consumption
    for the test period.

    Args:
        merged_df: DataFrame with columns ds, y (actual), yhat, yhat_lower, yhat_upper.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    # Confidence interval ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([merged_df["ds"], merged_df["ds"].iloc[::-1]]),
        y=pd.concat([merged_df["yhat_upper"], merged_df["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
        showlegend=True,
    ))

    # Actual line
    fig.add_trace(go.Scatter(
        x=merged_df["ds"],
        y=merged_df["y"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#00B4D8", width=2),
        marker=dict(size=4),
    ))

    # Predicted line
    fig.add_trace(go.Scatter(
        x=merged_df["ds"],
        y=merged_df["yhat"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#FF6B6B", width=2, dash="dot"),
        marker=dict(size=4),
    ))

    fig.update_layout(
        title=dict(text="📈 Actual vs Predicted — Test Period", font=dict(size=20)),
        xaxis_title="Date / Time",
        yaxis_title="Energy Consumption",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=40, r=20, t=70, b=40),
    )

    return fig


def plot_residuals(merged_df: pd.DataFrame) -> go.Figure:
    """
    Plot the residuals (actual - predicted) for the test period.

    Helps visualise systematic bias or heteroscedasticity.

    Args:
        merged_df: DataFrame with 'y' (actual) and 'yhat' (predicted) columns.

    Returns:
        A Plotly Figure object.
    """
    residuals = merged_df["y"] - merged_df["yhat"]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Residuals Over Time", "Residual Distribution"],
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(go.Scatter(
        x=merged_df["ds"],
        y=residuals,
        mode="lines+markers",
        name="Residuals",
        line=dict(color="#A8DADC"),
        marker=dict(size=3),
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals,
        name="Distribution",
        marker_color="#48CAE4",
        nbinsx=30,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        title=dict(text="🔍 Residual Analysis", font=dict(size=20)),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Peak Consumption Charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_top_peaks(future_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Bar chart of the top N predicted high-consumption periods.

    Args:
        future_df: Future forecast DataFrame (ds, yhat, yhat_lower, yhat_upper).
        top_n: Number of top periods to display.

    Returns:
        A Plotly Figure object.
    """
    top = future_df.nlargest(top_n, "yhat").sort_values("yhat", ascending=True)

    fig = go.Figure(go.Bar(
        x=top["yhat"],
        y=top["ds"].dt.strftime("%Y-%m-%d %H:%M"),
        orientation="h",
        marker=dict(
            color=top["yhat"],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="kWh"),
        ),
        text=top["yhat"].round(1),
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text=f"⚠️ Top {top_n} Predicted Peak Consumption Periods", font=dict(size=18)),
        xaxis_title="Predicted Consumption",
        yaxis_title="Date / Time",
        template="plotly_dark",
        height=420,
        margin=dict(l=160, r=40, t=60, b=40),
    )

    return fig
