import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
EMONCMS_URL = "https://emoncms.org"
API_KEY = "8fd531f2a88ab44e99029a9c68f6497a"  # Read-Only Key
FEED_IDS = {
    "voltage": 534451,
    "current": 534453,
    "power": 534454,
    "energy_kwh": 534455,
    "frequency": 534456,
    "pf": 534457
}
FEATURES = ["voltage", "current", "energy_kwh", "pf", "frequency"]
PARAMS = ["voltage", "current", "power", "energy_kwh", "frequency", "pf"]
TIMESTEPS = 5  # Match your LSTM training config
PESO_PER_KWH = 13  # Average rate (PHP)
ASSUMED_RECORDING_INTERVAL = 60  # Seconds

# --- LOAD MODELS (RF + LSTM) ---
@st.cache_resource
def load_models():
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    # Load BOTH models now
    rf = joblib.load("models/random_forest.pkl")
    lstm = load_model("models/lstm_model.keras")
    return scaler_X, scaler_y, rf, lstm

scaler_X, scaler_y, rf_model, lstm_model = load_models()

# --- 1. GET LIVE VALUES ---
def get_live_values():
    ids = ','.join(str(FEED_IDS[f]) for f in PARAMS)
    url = f"{EMONCMS_URL}/feed/fetch.json?ids={ids}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        values = response.json()
        return dict(zip(PARAMS, values))
    except:
        return {f: 0.0 for f in PARAMS}

# --- 2. GET HISTORY DATA ---
def get_history_for_feed(fid, start_time, end_time, interval):
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    url = f"{EMONCMS_URL}/feed/data.json?id={fid}&start={start_ms}&end={end_ms}&interval={interval}&skipmissing=1&limitinterval=1&apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if not data: return []
        return data
    except:
        return []

# --- 3. GET FULL HISTORICAL DATA ---
def get_full_history_data(start_date, end_date, interval):
    data = get_history_for_feed(FEED_IDS['power'], start_date, end_date, interval)
    if not data: return pd.DataFrame()
    
    times = [pd.to_datetime(d[0], unit='ms') for d in data]
    historical_data = {'power': [d[1] for d in data]}
    
    for feature in FEATURES:
        feat_data = get_history_for_feed(FEED_IDS[feature], start_date, end_date, interval)
        if len(feat_data) == len(data):
            historical_data[feature] = [d[1] for d in feat_data]
        else:
            mean_val = np.mean([d[1] for d in feat_data]) if feat_data else 0.0
            historical_data[feature] = [mean_val] * len(data)

    df = pd.DataFrame(historical_data, index=times)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df

# --- 4. PREDICT WITH RANDOM FOREST (Instant) ---
def predict_with_rf(features_dict, scaler_X, scaler_y):
    # Prepare input vector
    X = np.array([[features_dict[f] for f in FEATURES]])
    X_scaled = scaler_X.transform(X)
    # Predict
    y_scaled = rf_model.predict(X_scaled)
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
    return max(0, y)

# --- 5. FORECAST NEXT PERIOD WITH LSTM (Future Sequence) ---
def forecast_next_period_lstm(df_past, num_steps, interval_sec, scaler_X, scaler_y):
    if len(df_past) < TIMESTEPS:
        return pd.DataFrame()
    
    # Get initial state from the end of the past data
    last_row = df_past.iloc[-1]
    voltage_const = last_row['voltage']
    pf_const = last_row['pf']
    frequency_const = last_row['frequency']
    current_last = last_row['current']
    energy_last = last_row['energy_kwh']
    
    # Prepare the initial sequence for LSTM
    recent_features = df_past[FEATURES].tail(TIMESTEPS).values
    predictions = []
    current_time = df_past.index[-1] + timedelta(seconds=interval_sec)
    
    for _ in range(num_steps):
        # Scale and Reshape for LSTM
        X_scaled = scaler_X.transform(recent_features)
        X_lstm = X_scaled.reshape(1, TIMESTEPS, len(FEATURES))
        
        # Predict Power
        y_scaled = lstm_model.predict(X_lstm, verbose=0)
        pred_power = max(0, scaler_y.inverse_transform(y_scaled)[0][0])
        
        # Estimate next state of features
        interval_hours = interval_sec / 3600.0
        # P = V * I * PF -> I = P / (V * PF)
        new_current = pred_power / (voltage_const * pf_const) if voltage_const * pf_const > 0 else current_last
        new_energy = energy_last + (pred_power * interval_hours / 1000.0)
        
        new_row = np.array([voltage_const, new_current, new_energy, pf_const, frequency_const])
        
        # Update sliding window
        recent_features = np.vstack((recent_features[1:], new_row.reshape(1, -1)))
        
        predictions.append((current_time, pred_power))
        current_time += timedelta(seconds=interval_sec)
    
    df_pred = pd.DataFrame(predictions, columns=['time', 'predicted_power'])
    df_pred.set_index('time', inplace=True)
    return df_pred

# --- 6. FORECAST MONTHLY USAGE ---
def forecast_monthly(df_hist, duration_days, interval):
    interval_hours = interval / 3600.0
    total_kwh = (df_hist['power'] * interval_hours / 1000.0).sum()
    avg_daily_kwh = total_kwh / max(1, duration_days)
    monthly_kwh = avg_daily_kwh * 30
    monthly_peso = monthly_kwh * PESO_PER_KWH
    return monthly_kwh, monthly_peso

# --- DASHBOARD UI ---
st.set_page_config(page_title="Power Monitoring", page_icon="⚡", layout="wide")
st.title("⚡ Smart Home Energy Tracker")

# Sidebar
st.sidebar.header("Configuration")
use_live_data = st.sidebar.checkbox("Use Live Data", value=True)

st.sidebar.subheader("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=1))
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

duration_days = (end_date - start_date).days + 1
selected_range_label = f"{start_date} to {end_date} ({duration_days} days)"

# Dynamic interval for plotting
total_seconds = (datetime.combine(end_date, datetime.max.time()) - datetime.combine(start_date, datetime.min.time())).total_seconds()
interval = max(60, int(total_seconds / 100))

st.sidebar.subheader("Forecast Next Period (LSTM)")
enable_forecast = st.sidebar.checkbox("Enable Next Period Forecast", value=False)
if enable_forecast:
    forecast_period = st.sidebar.selectbox("Forecast Period", ["Daily", "Weekly", "Monthly", "Yearly"])
    past_start = st.sidebar.date_input("Start of Past Period", value=date(2025, 12, 1))
    
    if forecast_period == "Daily":
        past_end = past_start
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start
        forecast_interval = 3600
        period_label = "Day"
    elif forecast_period == "Weekly":
        past_end = past_start + timedelta(days=6)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=6)
        forecast_interval = 3600 * 4
        period_label = "Week"
    elif forecast_period == "Monthly":
        past_end = past_start + relativedelta(months=1, days=-1)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + relativedelta(months=1, days=-1)
        forecast_interval = 86400
        period_label = "Month"
    else:
        past_end = past_start + relativedelta(years=1, days=-1)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + relativedelta(years=1, days=-1)
        forecast_interval = 86400 * 30
        period_label = "Year"
    
    st.sidebar.info(f"Predicting next {forecast_period.lower()} ({forecast_start} to {forecast_end}) based on {past_start} to {past_end}")

st.sidebar.subheader("Filter Views")
show_params = st.sidebar.checkbox("Show Parameters", value=True)
show_predictions = st.sidebar.checkbox("Show Model Predictions (Graph)", value=False)
show_graphs = st.sidebar.checkbox("Show Graphs", value=True)
show_forecast = st.sidebar.checkbox("Show Monthly Forecast", value=True)

# Fetch Data
if use_live_data:
    live_values = get_live_values()
    features_dict = {k: v for k, v in live_values.items() if k in FEATURES}
    live_power = live_values.get("power", 0.0)
else:
    live_values = {}
    voltage = st.sidebar.slider("Voltage", 0.0, 300.0, 240.0)
    current = st.sidebar.slider("Current", 0.0, 10.0, 1.0)
    energy_kwh = st.sidebar.slider("Energy (kWh)", 0.0, 100.0, 15.0)
    pf = st.sidebar.slider("Power Factor", 0.0, 1.0, 0.8)
    frequency = st.sidebar.slider("Frequency", 50.0, 60.0, 60.0)
    live_power = 0.0
    features_dict = {
        "voltage": voltage, "current": current, "energy_kwh": energy_kwh,
        "pf": pf, "frequency": frequency
    }

# --- Historical Data Fetching ---
with st.spinner('Fetching historical data...'):
    df_hist = get_full_history_data(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), interval)

# --- Forecasting Data Fetching ---
df_past = pd.DataFrame()
df_actual_forecast = pd.DataFrame()
if enable_forecast:
    with st.spinner('Fetching past period data...'):
        df_past = get_full_history_data(
            datetime.combine(past_start, datetime.min.time()),
            datetime.combine(past_end, datetime.max.time()),
            forecast_interval
        )
    with st.spinner('Fetching actual forecast period data...'):
        df_actual_forecast = get_full_history_data(
            datetime.combine(forecast_start, datetime.min.time()),
            datetime.combine(forecast_end, datetime.max.time()),
            forecast_interval
        )
    if df_past.empty or df_actual_forecast.empty:
        st.warning("Insufficient data for forecasting. Check date range or API.")
        enable_forecast = False

# --- Instant Prediction (Random Forest) ---
# We use RF here because it's better for instantaneous mapping
pred_power = predict_with_rf(features_dict, scaler_X, scaler_y)

# --- Forecast Next Period (LSTM) ---
# We use LSTM here because it handles time-series sequences
df_forecast = pd.DataFrame()
if enable_forecast and not df_past.empty:
    num_steps = len(df_actual_forecast)
    with st.spinner('Generating forecast with LSTM...'):
        df_forecast = forecast_next_period_lstm(df_past, num_steps, forecast_interval, scaler_X, scaler_y)

# --- Display Parameters ---
if show_params:
    st.subheader("Current Parameters")
    cols = st.columns(3)
    for i, p in enumerate(PARAMS):
        value = live_values.get(p, 0.0) if use_live_data else features_dict.get(p, live_power if p == "power" else 0.0)
        cols[i % 3].metric(p.capitalize().replace("_kwh", " (kWh)"), f"{value:.2f}")

# --- User Recommendations ---
st.subheader("🤖 Analysis (RF Model)")
if pred_power > 1500:
    st.error("⚠️ **High Load Predicted:** Heavy appliances may be active (>1.5kW).")
elif pred_power > 200:
    st.warning("⚠️ **Baseload Alert:** Predicted usage >200W. Check devices.")
else:
    st.success("✅ **Efficient:** Predicted consumption is low.")

# --- Graphs & Analysis ---
if not df_hist.empty and show_graphs:
    st.subheader(f"📊 Historical Trends ({selected_range_label})")
    
    predicted_col_exists = False
    
    # Generate Historical Predictions for Graph using RF
    # RF is much faster for batch processing historical rows than LSTM
    if show_predictions:
        X = df_hist[FEATURES].values
        if len(X) > 0:
            X_scaled = scaler_X.transform(X)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            df_hist['predicted'] = [max(0, p) for p in y_pred]
            predicted_col_exists = True

    avg_pred = "N/A"
    if predicted_col_exists:
        avg_pred_val = df_hist['predicted'].mean(skipna=True)
        avg_pred = f"{avg_pred_val:.1f} W"
    st.metric("Avg Predicted Power (Range)", avg_pred)

    tab1, tab2, tab3, tab4 = st.tabs(["Power & Predictions (RF)", "Electrical Params", "Energy Consumed", "Period Forecast (LSTM)"])

    with tab1:
        st.markdown("### Power Consumption (Watts)")
        if predicted_col_exists:
            st.line_chart(df_hist[['power', 'predicted']])
        else:
            st.line_chart(df_hist[['power']])

    with tab2:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("### Voltage (V)")
            st.line_chart(df_hist['voltage'])
            st.markdown("### Power Factor")
            st.line_chart(df_hist['pf'])
        with col_g2:
            st.markdown("### Current (A)")
            st.line_chart(df_hist['current'])
            st.markdown("### Frequency (Hz)")
            st.line_chart(df_hist['frequency'])

    with tab3:
        st.markdown("### Cumulative Energy (kWh)")
        st.line_chart(df_hist['energy_kwh'])

    with tab4:
        if not df_forecast.empty:
            st.markdown(f"### 📉 Forecast Accuracy Analysis (Next {period_label}) - LSTM")
            
            # Align Data
            compare_df = df_actual_forecast[['power']].join(df_forecast, how='inner').dropna()
            compare_df.columns = ['Actual Power', 'Predicted Power']
            
            if not compare_df.empty:
                y_true = compare_df['Actual Power']
                y_pred = compare_df['Predicted Power']
                
                # Calculate Error Metrics
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                
                # MAPE (Handle division by zero)
                non_zero_mask = y_true != 0
                if np.sum(non_zero_mask) > 0:
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                else:
                    mape = 0.0

                # Display Metrics
                st.markdown("#### Performance Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Mean Abs. Error (MAE)", f"{mae:.2f} W", help="Avg error in Watts")
                m_col2.metric("RMSE", f"{rmse:.2f} W", help="Penalizes large spikes")
                m_col3.metric("Mean Abs. % Error (MAPE)", f"{mape:.2f} %", help="Avg % difference")

                st.line_chart(compare_df)
                
                with st.expander("View Detailed Data Table"):
                    compare_df['Error (W)'] = compare_df['Actual Power'] - compare_df['Predicted Power']
                    compare_df['% Error'] = (compare_df['Error (W)'] / compare_df['Actual Power']) * 100
                    st.dataframe(compare_df.style.format("{:.2f}"))
            else:
                st.warning("Timestamps do not align. Cannot calculate error.")
        else:
            st.info("Select dates and enable forecast to view.")

else:
    st.info("No history data available for the selected range.")

# --- Monthly Forecast ---
if show_forecast and not df_hist.empty:
    st.subheader("📅 Monthly Forecast")
    monthly_kwh, monthly_peso = forecast_monthly(df_hist, duration_days, interval)
    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Projected Monthly kWh", f"{monthly_kwh:.2f}")
    col_f2.metric("Projected Monthly Cost (PHP)", f"{monthly_peso:.2f}")
    st.info(f"Assumption: Usage repeats for 30 days at {PESO_PER_KWH} PHP/kWh.")