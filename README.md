# ⚡ Energy Consumption Forecaster

> **Time-Series Forecasting using Facebook Prophet** — A beginner-friendly yet professional portfolio project demonstrating energy demand prediction with an interactive Streamlit dashboard.

---

## 📸 Features

- 📂 **Upload any CSV** with a datetime and numeric energy column
- 🤖 **Prophet-powered forecasting** with automatic seasonality detection
- 📊 **Interactive Plotly charts** — historical, forecast, confidence intervals
- 🌀 **Seasonality decomposition** — trend, weekly, yearly, daily components
- 📉 **Model evaluation** — MAE, RMSE, MAPE on held-out 20% test data
- ⚡ **Peak consumption analysis** — KPI cards + top 10 peak periods
- ⬇️ **CSV download** of forecast results
- 🛡️ **Robust error handling** — friendly messages for bad data
- ⚡ **Streamlit caching** — Prophet only re-trains when data/config changes

---

## 🏗️ Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computations |
| **Prophet** | Time-series forecasting model |
| **Matplotlib** | Prophet's built-in component plots |
| **Plotly** | Interactive charts in Streamlit |
| **Streamlit** | Web dashboard framework |
| **Scikit-learn** | Evaluation metrics (MAE, RMSE) |

---

## 📁 Project Architecture

```
energy-consumption-forecaster/
│
├── app.py                        # Main Streamlit UI application
├── generate_sample_data.py       # Script to create sample CSVs
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore
│
├── data/
│   ├── sample_energy.csv         # 2 years of daily synthetic data
│   └── sample_energy_hourly.csv  # 60 days of hourly synthetic data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # CSV loading, column detection, validation
│   ├── preprocessing.py          # Frequency detection, train/test split
│   ├── forecasting.py            # Prophet model setup and prediction
│   ├── evaluation.py             # MAE, RMSE, MAPE metrics
│   └── visualization.py          # All Plotly + Matplotlib chart functions
│
└── models/
    └── .gitkeep                  # Directory for saved model artifacts
```

---

## 🚀 Quick Start

### Windows

```batch
:: 1. Navigate to the project folder
cd "path\to\energy-consumption-forecaster"

:: 2. Create virtual environment
python -m venv venv

:: 3. Activate virtual environment
venv\Scripts\activate

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. Generate sample datasets
python generate_sample_data.py

:: 6. Run the Streamlit application
streamlit run app.py
```

### Linux / macOS

```bash
# 1. Navigate to the project folder
cd path/to/energy-consumption-forecaster

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Generate sample datasets
python generate_sample_data.py

# 6. Run the Streamlit application
streamlit run app.py
```

The application will open automatically at **http://localhost:8501**.

---

## 📄 Dataset Format

Your CSV must contain at least two columns:

| Column | Description | Example |
|--------|-------------|---------|
| Date/Time | Parseable datetime string | `2024-01-01 00:00:00` |
| Energy | Numeric consumption value | `120.5` |

**Minimum example:**

```csv
timestamp,energy_consumption
2024-01-01 00:00:00,120.5
2024-01-01 01:00:00,118.7
2024-01-01 02:00:00,115.3
```

### Column naming
Column names are **automatically detected** — you do not need to name them `timestamp` and `energy_consumption`. The app will find the most likely date column and numeric column, or let you select them manually.

### Supported frequencies
- **Daily** — one row per day
- **Hourly** — one row per hour

---

## 🤖 How Prophet Works

[Facebook Prophet](https://facebook.github.io/prophet/) decomposes a time series into additive components:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + noise
```

### Components used in this project

| Component | Description | When enabled |
|-----------|-------------|--------------|
| **Trend** | Long-term growth or decline | Always |
| **Weekly seasonality** | How usage changes across days of the week | When ≥ 2 weeks of data |
| **Yearly seasonality** | Recurring annual cycles | When ≥ 2 years of data |
| **Daily seasonality** | Intra-day cycles (morning/evening peaks) | Hourly data with ≥ 48 observations |

### Why enable seasonality conditionally?
If you try to fit a yearly seasonality with only 3 months of data, Prophet cannot learn it reliably and will produce misleading results. This application checks data length before enabling each component.

---

## 📊 How Forecasting Works

1. **Upload CSV** → columns are detected automatically
2. **Data is cleaned**: invalid dates and values are removed, duplicates dropped, sorted chronologically
3. **DataFrame is converted to Prophet format**: `ds` (datetime) + `y` (numeric)
4. **Prophet model is configured** based on data frequency and length
5. **Model is fitted** on the data
6. **Future dataframe is created** for the selected horizon (7–90 days)
7. **Forecast is generated** with 95% confidence intervals
8. **Charts and metrics** are displayed in the dashboard

---

## 📉 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|actual − predicted\|) | Average absolute error in the same unit as energy |
| **RMSE** | √mean((actual − predicted)²) | More sensitive to large errors than MAE |
| **MAPE** | mean(\|actual − predicted\| / actual) × 100% | Error as a percentage of actual value |

A **chronological 80/20 train-test split** is used for evaluation — the model is trained on the first 80% and evaluated on the last 20%, preventing data leakage.

---

## 🗂️ How Each Major File Works

### `app.py`
The Streamlit entry point. Contains all UI logic: sidebar controls, tab layout, KPI cards, metric displays, and orchestration of the `src/` modules. Uses `@st.cache_data` and `@st.cache_resource` for performance.

### `src/data_loader.py`
- `load_csv()`: Reads any CSV from a file path or buffer.
- `detect_datetime_column()`: Auto-identifies the date column by name and parseability.
- `detect_value_column()`: Auto-identifies the numeric energy column.
- `prepare_prophet_dataframe()`: Cleans data and produces the `ds`/`y` DataFrame Prophet expects.
- `get_data_summary()`: Returns key stats for the Overview tab.

### `src/preprocessing.py`
- `detect_frequency()`: Determines hourly vs. daily by computing median time differences.
- `train_test_split()`: Creates a chronological split (no shuffling — this is time-series data!).
- `check_seasonality_feasibility()`: Decides which seasonalities can be reliably learned.

### `src/forecasting.py`
- `build_prophet_model()`: Configures Prophet with the right seasonality flags.
- `fit_model()`: Wraps `model.fit()` with error handling.
- `generate_forecast()`: Creates the future dataframe and calls `model.predict()`.
- `calculate_forecast_periods()`: Converts forecast days → number of hourly/daily periods.
- `extract_future_forecast()`: Strips historical rows from the Prophet output.

### `src/evaluation.py`
- `predict_test_period()`: Runs Prophet on the exact test-set timestamps.
- `compute_metrics()`: Calculates MAE, RMSE, MAPE using sklearn.
- `merge_actuals_and_predictions()`: Joins actual and predicted values for plotting.

### `src/visualization.py`
All chart functions. Plotly is used for interactive charts; Matplotlib for Prophet's native component plots. Charts use a consistent dark theme matching the Streamlit app.

---

## 💡 Example Workflow

1. Run `streamlit run app.py` — the app loads with sample data automatically.
2. Explore the **Overview** tab — check record count, date range, and detected seasonalities.
3. Go to **Historical Data** — adjust the rolling window slider to smooth the chart.
4. Sidebar: set **Forecast Horizon** to 30 days → click **Generate Forecast**.
5. **Forecast** tab — see the interactive chart; expand the Matplotlib chart below.
6. **Seasonality** tab — read the component plots (trend, weekly pattern, etc.).
7. **Model Evaluation** tab — view MAE/RMSE/MAPE and the actual vs. predicted chart.
8. **Forecast Data** tab — view the peak analysis and download the results CSV.

---

## 🔮 Possible Future Improvements

| Idea | Description |
|------|-------------|
| **Holiday effects** | Add country-specific holiday calendars via Prophet's `add_country_holidays()` |
| **Multiple regressors** | Include weather data (temperature, humidity) as additional Prophet regressors |
| **Cross-validation** | Use Prophet's built-in `cross_validation()` and `performance_metrics()` |
| **Anomaly detection** | Flag historical anomalies using residual z-scores |
| **Multi-location** | Support multiple energy meters / locations in one CSV |
| **Export model** | Serialize and save the fitted Prophet model with `joblib` |
| **Real dataset integration** | Connect to Kaggle's hourly energy consumption dataset |
| **Alerts** | Email or Slack alert when predicted consumption exceeds a threshold |

---

## ⚠️ Assumptions & Limitations

- The application assumes data is in a **regular time series** (no large gaps).
- Prophet is designed for **daily or sub-daily** data with at least several weeks of history.
- **Yearly seasonality** requires at least 2 full years of data for reliable results.
- The synthetic sample data is for demonstration only and does not represent real-world energy systems.
- Very short datasets (<20 rows) cannot be evaluated with the train/test split.

---

## 📝 License

This project is intended for educational and portfolio use. Feel free to adapt and extend it.
