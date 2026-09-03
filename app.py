"""
app.py
⚡ Energy Consumption Forecaster — Main Streamlit Application

Run with:
    streamlit run app.py
"""

import io
import warnings
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress verbose Prophet / cmdstanpy logs in the UI
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from src.data_loader import (
    load_csv,
    detect_datetime_column,
    detect_value_column,
    prepare_prophet_dataframe,
    get_data_summary,
)
from src.preprocessing import detect_frequency, train_test_split, check_seasonality_feasibility
from src.forecasting import (
    build_prophet_model,
    fit_model,
    generate_forecast,
    calculate_forecast_periods,
    extract_future_forecast,
)
from src.evaluation import predict_test_period, compute_metrics, merge_actuals_and_predictions
from src.visualization import (
    plot_historical,
    plot_daily_pattern,
    plot_monthly_pattern,
    plot_forecast_interactive,
    plot_forecast_matplotlib,
    plot_components_matplotlib,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_top_peaks,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⚡ Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(0, 180, 216, 0.2);
    }

    /* KPI metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,180,216,0.1) 0%, rgba(72,202,228,0.05) 100%);
        border: 1px solid rgba(0, 180, 216, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card h3 {
        color: #90E0EF;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
    }

    /* Alert / info banners */
    .info-banner {
        background: rgba(0, 180, 216, 0.08);
        border-left: 4px solid #00B4D8;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        color: rgba(255,255,255,0.85);
        font-size: 0.88rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #90E0EF;
        border-bottom: 1px solid rgba(0,180,216,0.25);
        padding-bottom: 8px;
        margin: 20px 0 14px 0;
    }

    /* Tab overrides */
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00B4D8, #0077B6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Generate forecast button */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00B4D8, #023E8A) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px !important;
        margin-top: 6px;
    }

    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #48CAE4, #0077B6) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4) !important;
    }

    /* Streamlit metric blocks */
    [data-testid="metric-container"] {
        background: rgba(0,180,216,0.07);
        border: 1px solid rgba(0,180,216,0.25);
        border-radius: 10px;
        padding: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str = "") -> str:
    """Return an HTML KPI card string."""
    return f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


def info_banner(text: str) -> None:
    st.markdown(f'<div class="info-banner">{text}</div>', unsafe_allow_html=True)


def section_header(text: str) -> None:
    st.markdown(f'<p class="section-header">{text}</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_prepare_dataframe(raw_csv_bytes: bytes, datetime_col: str, value_col: str):
    """Cache the cleaned Prophet DataFrame to avoid reprocessing on every rerun."""
    import io
    raw_df = load_csv(io.BytesIO(raw_csv_bytes))
    prophet_df, warnings_list = prepare_prophet_dataframe(raw_df, datetime_col, value_col)
    return prophet_df, warnings_list


@st.cache_resource(show_spinner=False)
def cached_fit_model(train_key: str, freq: str, seasonality: dict, yearly_override: bool, train_df_json: str):
    """Cache fitted Prophet model. Key is a hash of training data + config."""
    train_df = pd.read_json(io.StringIO(train_df_json), convert_dates=["ds"])
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    model = build_prophet_model(freq, seasonality, yearly_override)
    model = fit_model(model, train_df)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size:3rem; font-weight:800; background: linear-gradient(135deg, #00B4D8, #90E0EF);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin-bottom:0.3rem;">⚡ Energy Consumption Forecaster</h1>
    <p style="color:rgba(255,255,255,0.55); font-size:1.1rem; margin:0;">
        Time-Series Forecasting using Facebook Prophet
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload CSV Dataset",
        type=["csv"],
        help="Upload an energy consumption CSV file. Needs at least a date column and a numeric column.",
    )

    if uploaded_file is None:
        sample_path = "data/sample_energy.csv"
        try:
            with open(sample_path, "rb") as f:
                sample_bytes = f.read()
            st.info("No file uploaded. Using **sample_energy.csv** by default.")
            uploaded_file_bytes = sample_bytes
            uploaded_file_name = "sample_energy.csv"
            use_sample = True
        except FileNotFoundError:
            uploaded_file_bytes = None
            uploaded_file_name = None
            use_sample = False
    else:
        uploaded_file_bytes = uploaded_file.read()
        uploaded_file_name = uploaded_file.name
        use_sample = False

    st.divider()

    # Column selectors (only shown after loading a file)
    datetime_col_select = None
    value_col_select = None
    forecast_days = 30
    freq_choice = "Auto-detect"
    show_confidence = True
    yearly_override = False
    generate_clicked = False

    if uploaded_file_bytes:
        try:
            raw_preview = load_csv(io.BytesIO(uploaded_file_bytes))
            all_cols = list(raw_preview.columns)

            auto_dt = detect_datetime_column(raw_preview) or all_cols[0]
            auto_val = detect_value_column(raw_preview, auto_dt) or all_cols[1]

            datetime_col_select = st.selectbox(
                "🗓️ Date / Time Column",
                options=all_cols,
                index=all_cols.index(auto_dt) if auto_dt in all_cols else 0,
            )
            value_col_select = st.selectbox(
                "⚡ Energy Consumption Column",
                options=[c for c in all_cols if c != datetime_col_select],
                index=max(all_cols.index(auto_val) - 1, 0) if auto_val in all_cols else 0,
            )

            st.divider()

            forecast_days = st.selectbox(
                "🔮 Forecast Horizon (days)",
                options=[7, 14, 30, 60, 90],
                index=2,
            )

            freq_choice = st.radio(
                "📡 Data Frequency",
                options=["Auto-detect", "Daily", "Hourly"],
                index=0,
            )

            show_confidence = st.checkbox("📊 Show Confidence Interval", value=True)
            yearly_override = st.checkbox(
                "📅 Force Yearly Seasonality",
                value=False,
                help="Enable this only if you believe your dataset has yearly patterns but spans < 2 years.",
            )

            st.divider()

            generate_clicked = st.button("🚀 Generate Forecast", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Could not preview file: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main content — only shown when data is available
# ──────────────────────────────────────────────────────────────────────────────

if not uploaded_file_bytes:
    # Landing / empty state
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:rgba(255,255,255,0.4);">
        <div style="font-size:5rem;">📂</div>
        <h3 style="color:rgba(255,255,255,0.6);">Upload a CSV to Get Started</h3>
        <p>Use the sidebar to upload your energy consumption dataset,<br>
        or the application will automatically load the built-in sample data.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Load and prepare data
# ──────────────────────────────────────────────────────────────────────────────

try:
    with st.spinner("🔄 Loading and cleaning dataset..."):
        prophet_df, data_warnings = cached_prepare_dataframe(
            uploaded_file_bytes, datetime_col_select, value_col_select
        )
except ValueError as e:
    st.error(f"❌ Data Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error while loading data: {e}")
    st.stop()

# Show warnings
for w in data_warnings:
    st.warning(w)

# Detect or override frequency
if freq_choice == "Hourly":
    freq = "h"
elif freq_choice == "Daily":
    freq = "D"
else:
    freq = detect_frequency(prophet_df)

# Summary stats
summary = get_data_summary(prophet_df)

# Seasonality feasibility
seasonality = check_seasonality_feasibility(prophet_df, freq)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏠 Overview",
    "📊 Historical Data",
    "🔮 Forecast",
    "🌀 Seasonality",
    "📉 Model Evaluation",
    "📋 Forecast Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📋 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.metric("📦 Records", f"{summary['num_records']:,}")
    with col2:
        st.metric("📅 Start Date", summary["start_date"].strftime("%Y-%m-%d"))
    with col3:
        st.metric("📅 End Date", summary["end_date"].strftime("%Y-%m-%d"))
    with col4:
        st.metric("⚡ Avg Consumption", f"{summary['avg_consumption']:.2f}")
    with col5:
        st.metric("⬇️ Min Consumption", f"{summary['min_consumption']:.2f}")
    with col6:
        st.metric("⬆️ Max Consumption", f"{summary['max_consumption']:.2f}")

    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📆 Date Range", f"{summary['date_range_days']} days")
    with col_b:
        st.metric("📡 Detected Frequency", summary["estimated_frequency"])
    with col_c:
        st.metric("🗓️ Forecast Frequency", "Hourly" if freq == "h" else "Daily")

    st.markdown("---")
    section_header("🔍 Seasonality Feasibility")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        icon = "✅" if seasonality["daily"] else "❌"
        st.metric("Daily Seasonality", f"{icon} {'Feasible' if seasonality['daily'] else 'Not Feasible'}")
    with s_col2:
        icon = "✅" if seasonality["weekly"] else "❌"
        st.metric("Weekly Seasonality", f"{icon} {'Feasible' if seasonality['weekly'] else 'Not Feasible'}")
    with s_col3:
        icon = "✅" if (seasonality["yearly"] or yearly_override) else "❌"
        st.metric("Yearly Seasonality", f"{icon} {'Feasible' if (seasonality['yearly'] or yearly_override) else 'Not Feasible'}")

    if not seasonality["yearly"] and not yearly_override:
        info_banner(
            "💡 Yearly seasonality is disabled because the dataset spans less than 2 full years. "
            "Enable <strong>Force Yearly Seasonality</strong> in the sidebar to override this."
        )

    st.markdown("---")
    section_header("📄 Raw Data Preview")
    display_df = prophet_df.copy()
    display_df.columns = ["Timestamp", "Energy Consumption"]
    st.dataframe(display_df.head(100), use_container_width=True, height=300)
    if len(prophet_df) > 100:
        st.caption(f"Showing first 100 of {len(prophet_df):,} rows.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historical Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("📊 Historical Energy Consumption")

    # Rolling window choice
    if freq == "h":
        default_window = 24
        window_label = "Rolling window (hours)"
    else:
        default_window = 7
        window_label = "Rolling window (days)"

    rolling_window = st.slider(window_label, min_value=1, max_value=90, value=default_window)

    fig_hist = plot_historical(prophet_df, rolling_window=rolling_window)
    st.plotly_chart(fig_hist, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_day = plot_daily_pattern(prophet_df)
        if fig_day:
            st.plotly_chart(fig_day, use_container_width=True)
        else:
            st.info("Not enough data to compute a daily pattern (need ≥ 14 rows).")

    with col_right:
        fig_month = plot_monthly_pattern(prophet_df)
        if fig_month:
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("Not enough data to compute a monthly pattern (need data spanning ≥ 2 months).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if not generate_clicked:
        st.markdown("""
        <div style="text-align:center; padding:3rem 2rem; color:rgba(255,255,255,0.4);">
            <div style="font-size:4rem;">🚀</div>
            <h3 style="color:rgba(255,255,255,0.6);">Ready to Forecast</h3>
            <p>Configure your settings in the sidebar and click <strong>Generate Forecast</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            with st.spinner("⚙️ Training Prophet model... this may take a moment."):
                periods = calculate_forecast_periods(forecast_days, freq)
                train_df_json = prophet_df.to_json(date_format="iso")
                train_key = f"{hash(train_df_json)}_{freq}_{yearly_override}"

                # Fit full model on ALL data for future forecasting
                model_full = cached_fit_model(
                    train_key, freq, seasonality, yearly_override, train_df_json
                )
                forecast_full = generate_forecast(model_full, periods, freq, include_history=True)

            train_end_date = prophet_df["ds"].max()
            future_df = extract_future_forecast(forecast_full, train_end_date)

            st.success(f"✅ Forecast generated: **{len(future_df):,}** future {'hours' if freq == 'h' else 'days'} predicted.")

            # Interactive Plotly forecast chart
            fig_fc = plot_forecast_interactive(prophet_df, forecast_full, show_confidence)
            st.plotly_chart(fig_fc, use_container_width=True)

            # Prophet Matplotlib chart
            with st.expander("📐 Prophet's Built-in Forecast Chart (Matplotlib)"):
                fig_mpl = plot_forecast_matplotlib(model_full, forecast_full)
                st.pyplot(fig_mpl, use_container_width=True)
                plt.close(fig_mpl)

            # Store in session state for other tabs
            st.session_state["model_full"] = model_full
            st.session_state["forecast_full"] = forecast_full
            st.session_state["future_df"] = future_df
            st.session_state["freq"] = freq
            st.session_state["prophet_df"] = prophet_df
            st.session_state["forecast_ready"] = True
            st.session_state["seasonality"] = seasonality
            st.session_state["yearly_override"] = yearly_override

        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Seasonality
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("🌀 Seasonality & Trend Components")

    if not st.session_state.get("forecast_ready"):
        st.info("⬅️ Generate a forecast first (Sidebar → **Generate Forecast**).")
    else:
        model_full = st.session_state["model_full"]
        forecast_full = st.session_state["forecast_full"]

        info_banner("""
            <strong>How to read the component plots:</strong><br>
            • <strong>Trend</strong>: Long-term direction of energy consumption (rising, falling, or flat).<br>
            • <strong>Weekly Seasonality</strong>: How consumption typically varies across the days of the week.<br>
            • <strong>Yearly Seasonality</strong>: Recurring annual patterns (e.g., higher usage in summer/winter).<br>
            • <strong>Daily Seasonality</strong>: Intra-day cycles (only visible for hourly data).
        """)

        with st.spinner("Rendering component charts..."):
            fig_comp = plot_components_matplotlib(model_full, forecast_full)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close(fig_comp)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("📉 Model Evaluation — Train / Test Split")

    if not st.session_state.get("forecast_ready"):
        st.info("⬅️ Generate a forecast first (Sidebar → **Generate Forecast**).")
    else:
        prophet_df_eval = st.session_state["prophet_df"]
        freq_eval = st.session_state["freq"]
        seasonality_eval = st.session_state["seasonality"]
        yearly_override_eval = st.session_state["yearly_override"]

        try:
            with st.spinner("⚙️ Running evaluation on train/test split..."):
                train_df, test_df = train_test_split(prophet_df_eval, test_fraction=0.2)

                train_key_eval = f"{hash(train_df.to_json(date_format='iso'))}_{freq_eval}_{yearly_override_eval}_eval"
                train_df_json_eval = train_df.to_json(date_format="iso")

                model_eval = cached_fit_model(
                    train_key_eval, freq_eval, seasonality_eval, yearly_override_eval, train_df_json_eval
                )
                forecast_test = predict_test_period(model_eval, test_df)
                merged_df = merge_actuals_and_predictions(test_df, forecast_test)
                metrics = compute_metrics(merged_df["y"], merged_df["yhat"])

            # Metrics info
            info_banner(
                f"Trained on <strong>{len(train_df):,}</strong> observations "
                f"(80%) and evaluated on the last <strong>{len(test_df):,}</strong> observations (20%)."
            )

            # Metric KPI cards
            st.markdown("#### Evaluation Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="MAE — Mean Absolute Error",
                    value=f"{metrics['MAE']:.3f}",
                    help="Average absolute difference between actual and predicted values. Lower is better.",
                )
            with m2:
                st.metric(
                    label="RMSE — Root Mean Squared Error",
                    value=f"{metrics['RMSE']:.3f}",
                    help="Penalises large errors more heavily than MAE. Lower is better.",
                )
            with m3:
                mape_str = f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics["MAPE"]) else "N/A"
                st.metric(
                    label="MAPE — Mean Abs % Error",
                    value=mape_str,
                    help="Average percentage error. Lower is better. N/A if actuals contain zeros.",
                )

            st.markdown("---")
            info_banner("""
                <strong>Metric Explanations:</strong><br>
                • <strong>MAE</strong>: On average, the model's prediction is off by this many units.<br>
                • <strong>RMSE</strong>: Similar to MAE but more sensitive to large outlier errors.<br>
                • <strong>MAPE</strong>: Expresses error as a percentage of the actual value — useful for comparing across scales.
            """)

            # Actual vs Predicted chart
            st.markdown("---")
            fig_avp = plot_actual_vs_predicted(merged_df)
            st.plotly_chart(fig_avp, use_container_width=True)

            # Residual analysis
            fig_res = plot_residuals(merged_df)
            st.plotly_chart(fig_res, use_container_width=True)

        except ValueError as e:
            st.error(f"❌ Evaluation Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected evaluation error: {e}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Forecast Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("📋 Forecast Results Table & Downloads")

    if not st.session_state.get("forecast_ready"):
        st.info("⬅️ Generate a forecast first (Sidebar → **Generate Forecast**).")
    else:
        future_df = st.session_state["future_df"]
        freq_tab = st.session_state["freq"]

        # ── Peak consumption analysis ────────────────────────────────────────
        section_header("⚡ Peak Consumption Analysis")

        peak_max_val = future_df["yhat"].max()
        peak_max_dt = future_df.loc[future_df["yhat"].idxmax(), "ds"]
        peak_min_val = future_df["yhat"].min()
        peak_avg_val = future_df["yhat"].mean()

        pk1, pk2, pk3, pk4 = st.columns(4)
        with pk1:
            st.metric("🔺 Peak Predicted", f"{peak_max_val:.2f}")
        with pk2:
            dt_fmt = "%Y-%m-%d %H:%M" if freq_tab.lower() == "h" else "%Y-%m-%d"
            st.metric("📅 Peak Date", peak_max_dt.strftime(dt_fmt))
        with pk3:
            st.metric("🔻 Min Predicted", f"{peak_min_val:.2f}")
        with pk4:
            st.metric("📊 Avg Predicted", f"{peak_avg_val:.2f}")

        st.markdown("")
        fig_peaks = plot_top_peaks(future_df, top_n=10)
        st.plotly_chart(fig_peaks, use_container_width=True)

        # ── Forecast table ────────────────────────────────────────────────────
        st.markdown("---")
        section_header("📄 Full Forecast Table")

        display_future = future_df.copy()
        display_future["ds"] = display_future["ds"].dt.strftime(
            "%Y-%m-%d %H:%M" if freq_tab.lower() == "h" else "%Y-%m-%d"
        )
        display_future = display_future.rename(columns={
            "ds": "Date / Time",
            "yhat": "Predicted",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound",
            "trend": "Trend",
        })
        display_future[["Predicted", "Lower Bound", "Upper Bound", "Trend"]] = (
            display_future[["Predicted", "Lower Bound", "Upper Bound", "Trend"]].round(3)
        )

        st.dataframe(display_future, use_container_width=True, height=400)

        # ── Download button ───────────────────────────────────────────────────
        csv_bytes = display_future.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=csv_bytes,
            file_name="energy_forecast.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.caption(f"Total future periods: **{len(future_df):,}** ({'hourly' if freq_tab.lower() == 'h' else 'daily'})")
