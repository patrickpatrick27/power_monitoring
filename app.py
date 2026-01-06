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
TIMESTEPS = 5  # Match your training config
PESO_PER_KWH = 13  # Average rate from Philippines (as of December 2025)
ASSUMED_RECORDING_INTERVAL = 60  # Seconds; adjust to your hardware's actual posting interval (e.g., 10-60s)

# --- LOAD MODELS (Cached for performance) ---
@st.cache_resource
def load_models():
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    rf = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    lstm = load_model("models/lstm_model.keras")
    return scaler_X, scaler_y, rf, xgb, lstm

scaler_X, scaler_y, rf_model, xgb_model, lstm_model = load_models()

# --- 1. GET LIVE VALUES FOR ALL FEATURES + POWER ---
def get_live_values():
    ids = ','.join(str(FEED_IDS[f]) for f in PARAMS)
    url = f"{EMONCMS_URL}/feed/fetch.json?ids={ids}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        values = response.json()
        return dict(zip(PARAMS, values))
    except:
        return {f: 0.0 for f in PARAMS}

# --- 2. GET HISTORY DATA FOR A SINGLE FEED ---
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

# --- 3. GET RECENT DATA FOR LSTM (Last TIMESTEPS points for features) ---
def get_recent_data(timesteps=TIMESTEPS):
    data = {}
    end_time = datetime.now()
    # Fix: Calculate a recent start time to cover at least TIMESTEPS points + buffer
    buffer = 2  # Extra points to ensure we get enough
    start_time = end_time - timedelta(seconds=ASSUMED_RECORDING_INTERVAL * (timesteps + buffer))
    for feature in FEATURES:
        fid = FEED_IDS[feature]
        # Use calculated start/end and fixed interval to get raw/recent points (no dp=; dp causes full-history sampling)
        feat_data = get_history_for_feed(fid, start_time, end_time, ASSUMED_RECORDING_INTERVAL)
        if feat_data:
            data[feature] = [d[1] for d in feat_data][-timesteps:]  # Take last timesteps
        else:
            data[feature] = [0.0] * timesteps

    df = pd.DataFrame(data)
    if len(df) < timesteps:
        # Pad with mean if too few
        pad = pd.DataFrame({f: [df[f].mean() if len(df) > 0 else 0.0] * (timesteps - len(df)) for f in FEATURES})
        df = pd.concat([pad, df], ignore_index=True)
    return df

# --- 4. GET FULL HISTORICAL DATA FOR ALL FEATURES ---
def get_full_history_data(start_date, end_date, interval):
    # Fetch Power first as the index reference
    data = get_history_for_feed(FEED_IDS['power'], start_date, end_date, interval)
    if not data: return pd.DataFrame()
    
    times = [pd.to_datetime(d[0], unit='ms') for d in data]
    historical_data = {'power': [d[1] for d in data]}
    
    # Fetch other features
    for feature in FEATURES:
        feat_data = get_history_for_feed(FEED_IDS[feature], start_date, end_date, interval)
        if len(feat_data) == len(data):
            historical_data[feature] = [d[1] for d in feat_data]
        else:
            st.warning(f"Data sync mismatch for {feature}. Using mean padding.")
            mean_val = np.mean([d[1] for d in feat_data]) if feat_data else 0.0
            historical_data[feature] = [mean_val] * len(data)  # Fix: Use mean instead of 0 for better robustness

    df = pd.DataFrame(historical_data, index=times)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df

# --- 5. PREDICT WITH TREE MODELS (RF/XGB) ---
def predict_with_tree(model, features_dict, scaler_X, scaler_y):
    X = np.array([[features_dict[f] for f in FEATURES]])
    X_scaled = scaler_X.transform(X)
    y_scaled = model.predict(X_scaled)
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
    return max(0, y)

# --- 6. PREDICT WITH LSTM (Next step forecast) ---
def predict_with_lstm(recent_df, scaler_X, scaler_y):
    X = recent_df.values
    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape(1, TIMESTEPS, len(FEATURES))
    y_scaled = lstm_model.predict(X_lstm, verbose=0)
    y = scaler_y.inverse_transform(y_scaled)[0][0]
    return max(0, y)

# --- 6.5 FORECAST NEXT PERIOD WITH LSTM ---
def forecast_next_period_lstm(df_past, num_steps, interval_sec, scaler_X, scaler_y):
    if len(df_past) < TIMESTEPS:
        return pd.DataFrame()  # Not enough data
    
    # Constants from last past data point
    last_row = df_past.iloc[-1]
    voltage_const = last_row['voltage']
    pf_const = last_row['pf']
    frequency_const = last_row['frequency']
    current_last = last_row['current']
    energy_last = last_row['energy_kwh']
    
    # Start with last TIMESTEPS features
    recent_features = df_past[FEATURES].tail(TIMESTEPS).values
    
    predictions = []
    current_time = df_past.index[-1] + timedelta(seconds=interval_sec)
    
    for _ in range(num_steps):
        # Predict next power
        X_scaled = scaler_X.transform(recent_features)
        X_lstm = X_scaled.reshape(1, TIMESTEPS, len(FEATURES))
        y_scaled = lstm_model.predict(X_lstm, verbose=0)
        pred_power = max(0, scaler_y.inverse_transform(y_scaled)[0][0])
        
        # Update features for next step
        interval_hours = interval_sec / 3600.0
        new_current = pred_power / (voltage_const * pf_const) if voltage_const * pf_const > 0 else current_last
        new_energy = energy_last + (pred_power * interval_hours / 1000.0)
        new_row = np.array([voltage_const, new_current, new_energy, pf_const, frequency_const])
        
        # Slide window: Append new row, remove oldest
        recent_features = np.vstack((recent_features[1:], new_row.reshape(1, -1)))  # Fix: Ensure new_row is 2D
        
        # Update lasts
        current_last = new_current
        energy_last = new_energy
        
        # Store
        predictions.append((current_time, pred_power))
        current_time += timedelta(seconds=interval_sec)
    
    df_pred = pd.DataFrame(predictions, columns=['time', 'predicted_power'])
    df_pred.set_index('time', inplace=True)
    return df_pred

# --- 7. FORECAST MONTHLY USAGE & COST ---
def forecast_monthly(df_hist, duration_days, interval):
    interval_hours = interval / 3600.0
    total_kwh = (df_hist['power'] * interval_hours / 1000.0).sum()
    avg_daily_kwh = total_kwh / max(1, duration_days)
    monthly_kwh = avg_daily_kwh * 30  # Approximate month
    monthly_peso = monthly_kwh * PESO_PER_KWH
    return monthly_kwh, monthly_peso

# --- DASHBOARD UI (All in one page) ---
st.set_page_config(page_title="Power Monitoring", page_icon="⚡", layout="wide")
st.title("⚡ Smart Home Energy Tracker")

# Sidebar for Interactivity & Filters
st.sidebar.header("Interact & Predict")
selected_model = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"])
use_live_data = st.sidebar.checkbox("Use Live Data", value=True)

st.sidebar.subheader("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=1))
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

duration_days = (end_date - start_date).days + 1
selected_range_label = f"{start_date} to {end_date} ({duration_days} days)"

# Dynamic interval for main history
total_seconds = (datetime.combine(end_date, datetime.max.time()) - datetime.combine(start_date, datetime.min.time())).total_seconds()
interval = max(60, int(total_seconds / 100))

st.sidebar.subheader("Forecast Next Period")
enable_forecast = st.sidebar.checkbox("Enable Next Period Forecast", value=False)
if enable_forecast:
    forecast_period = st.sidebar.selectbox("Forecast Period", ["Daily", "Weekly", "Monthly", "Yearly"])
    past_start = st.sidebar.date_input("Start of Past Period", value=date(2025, 12, 1))
    
    # Calculate past_end and forecast dates based on period
    if forecast_period == "Daily":
        past_end = past_start
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start
        forecast_interval = 3600  # Hourly for day (24 steps)
        period_label = "Day"
    elif forecast_period == "Weekly":
        past_end = past_start + timedelta(days=6)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=6)
        forecast_interval = 3600 * 4  # 4-hourly for week (~42 steps)
        period_label = "Week"
    elif forecast_period == "Monthly":
        past_end = past_start + relativedelta(months=1, days=-1)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + relativedelta(months=1, days=-1)
        forecast_interval = 86400  # Daily for month (~30 steps)
        period_label = "Month"
    else:  # Yearly
        past_end = past_start + relativedelta(years=1, days=-1)
        forecast_start = past_end + timedelta(days=1)
        forecast_end = forecast_start + relativedelta(years=1, days=-1)
        forecast_interval = 86400 * 30  # Monthly for year (~12 steps)
        period_label = "Year"
    
    st.sidebar.info(f"Predicting next {forecast_period.lower()} ({forecast_start} to {forecast_end}) based on {past_start} to {past_end}")

st.sidebar.subheader("Filter Views")
show_params = st.sidebar.checkbox("Show Parameters", value=True)
show_predictions = st.sidebar.checkbox("Show Model Predictions", value=False)
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
    live_power = 0.0  # Hypothetical, no live power
    features_dict = {
        "voltage": voltage, "current": current, "energy_kwh": energy_kwh,
        "pf": pf, "frequency": frequency
    }

with st.spinner('Fetching historical data...'):
    df_hist = get_full_history_data(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), interval)

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
        enable_forecast = False  # Disable to avoid errors

if use_live_data and selected_model == "LSTM":
    recent_df = get_recent_data()
else:
    recent_df = pd.DataFrame([features_dict] * TIMESTEPS) if not use_live_data else None

# --- Predictions (Single Point) ---
if selected_model == "Random Forest":
    pred_power = predict_with_tree(rf_model, features_dict, scaler_X, scaler_y)
elif selected_model == "XGBoost":
    pred_power = predict_with_tree(xgb_model, features_dict, scaler_X, scaler_y)
else:
    pred_power = predict_with_lstm(recent_df, scaler_X, scaler_y)

# --- Forecast Next Period ---
df_forecast = pd.DataFrame()
if enable_forecast and not df_past.empty:
    num_steps = len(df_actual_forecast)  # Match actual points
    with st.spinner('Generating forecast...'):
        df_forecast = forecast_next_period_lstm(df_past, num_steps, forecast_interval, scaler_X, scaler_y)

# --- Display 6 Parameters (if selected) ---
if show_params:
    st.subheader("Current Parameters")
    cols = st.columns(3)
    for i, p in enumerate(PARAMS):
        value = live_values.get(p, 0.0) if use_live_data else features_dict.get(p, live_power if p == "power" else 0.0)
        cols[i % 3].metric(p.capitalize().replace("_kwh", " (kWh)"), f"{value:.2f}")

# --- User Recommendations ---
st.subheader("🤖 User Recommendations")
if pred_power > 1500:
    st.error("⚠️ **High Load Predicted:** Heavy appliances may be active (>1.5kW).")
elif pred_power > 200:
    st.warning("⚠️ **Baseload Alert:** Predicted usage >200W. Check devices.")
else:
    st.success("✅ **Efficient:** Predicted consumption is low.")

# --- Historical Trends & Graphs (if selected) ---
if not df_hist.empty and show_graphs:
    st.subheader(f"📊 Historical Trends ({selected_range_label})")
    
    # Calculate Predictions if Requested
    predicted_col_exists = False
    if show_predictions:
        if selected_model in ["Random Forest", "XGBoost"]:
            model = rf_model if selected_model == "Random Forest" else xgb_model
            X = df_hist[FEATURES].values
            if len(X) > 0:
                X_scaled = scaler_X.transform(X)
                y_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
                df_hist['predicted'] = [max(0, p) for p in y_pred]
                predicted_col_exists = True
        else:  # LSTM
            X = df_hist[FEATURES].values
            if len(X) >= TIMESTEPS:
                X_scaled = scaler_X.transform(X)
                Xs = np.array([X_scaled[i:i + TIMESTEPS] for i in range(len(X_scaled) - TIMESTEPS)])
                y_scaled = lstm_model.predict(Xs, verbose=0)
                y_pred = scaler_y.inverse_transform(y_scaled).flatten()
                df_hist['predicted'] = np.nan
                df_hist.iloc[TIMESTEPS:, df_hist.columns.get_loc('predicted')] = [max(0, p) for p in y_pred]
                predicted_col_exists = True

    # Avg Predicted
    avg_pred = "N/A"
    if predicted_col_exists:
        avg_pred_val = df_hist['predicted'].mean(skipna=True)
        avg_pred = f"{avg_pred_val:.1f} W"

    st.metric("Avg Predicted Power (Range)", avg_pred)

    # Graphs in Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Power & Predictions", "Electrical Params", "Energy Consumed", "Period Forecast"])

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
            st.markdown(f"### Actual vs Predicted Power (Next {period_label})")
            compare_df = df_actual_forecast[['power']].join(df_forecast, how='outer')
            compare_df.columns = ['Actual Power', 'Predicted Power']
            st.line_chart(compare_df)
        else:
            st.info("Select dates and enable forecast to view.")

else:
    st.info("No history data available for the selected range.")

# --- Monthly Forecast & Cost (if selected) ---
if show_forecast and not df_hist.empty:
    st.subheader("📅 Monthly Forecast")
    monthly_kwh, monthly_peso = forecast_monthly(df_hist, duration_days, interval)
    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Projected Monthly kWh", f"{monthly_kwh:.2f}")
    col_f2.metric("Projected Monthly Cost (PHP)", f"{monthly_peso:.2f}")
    st.info(f"Assumption: Usage over selected {duration_days} days repeats for 30-day month at {PESO_PER_KWH} PHP/kWh (current avg rate).")

st.info("Tip: Use checkboxes in sidebar to filter views. For forecasting, select a period to extrapolate.")