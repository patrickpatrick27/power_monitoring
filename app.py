import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib
from datetime import datetime, timedelta, date
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
PESO_PER_KWH = 11.5  # Average rate from Philippines Dec 2024 (update as needed; sources: Meralco ~11.5-12 PHP/kWh)

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
    for feature in FEATURES:
        fid = FEED_IDS[feature]
        url = f"{EMONCMS_URL}/feed/data.json?id={fid}&start=0&end={int(datetime.now().timestamp()*1000)}&dp={timesteps}&apikey={API_KEY}"
        try:
            resp = requests.get(url, timeout=5).json()
            data[feature] = [d[1] for d in resp][-timesteps:]
        except:
            data[feature] = [0.0] * timesteps
    df = pd.DataFrame(data)
    if len(df) < timesteps:
        pad = pd.DataFrame({f: [df[f].mean()] * (timesteps - len(df)) for f in FEATURES})
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
            st.warning(f"Data sync mismatch for {feature}. Graph may be partial.")
            historical_data[feature] = [0] * len(data)

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

# --- 7. FORECAST MONTHLY USAGE & COST ---
def forecast_monthly(df_hist, duration_days, interval):
    interval_hours = interval / 3600.0
    total_kwh = (df_hist['power'] * interval_hours / 1000.0).sum()
    avg_daily_kwh = total_kwh / max(1, duration_days)
    monthly_kwh = avg_daily_kwh * 30  # Approximate month
    monthly_peso = monthly_kwh * PESO_PER_KWH
    return monthly_kwh, monthly_peso

# --- DASHBOARD UI (All in one page) ---
st.set_page_config(page_title="Power AI", page_icon="⚡", layout="wide")
st.title("⚡ Smart Energy Brain")

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

# Dynamic interval
total_seconds = (datetime.combine(end_date, datetime.max.time()) - datetime.combine(start_date, datetime.min.time())).total_seconds()
interval = max(60, int(total_seconds / 100))

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

# --- Display 6 Parameters (if selected) ---
if show_params:
    st.subheader("Current Parameters")
    cols = st.columns(3)
    for i, p in enumerate(PARAMS):
        value = live_values.get(p, 0.0) if use_live_data else features_dict.get(p, live_power if p == "power" else 0.0)
        cols[i % 3].metric(p.capitalize().replace("_kwh", " (kWh)"), f"{value:.2f}")

# --- AI Recommendations ---
st.subheader("🤖 AI Recommendations")
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
    tab1, tab2, tab3 = st.tabs(["Power & Predictions", "Electrical Params", "Energy Consumed"])

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

st.info("Tip: Use checkboxes in sidebar to filter views. For forecasting, select a week to extrapolate to month.")