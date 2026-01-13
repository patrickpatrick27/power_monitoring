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

# --- LOAD MODELS (RF + LSTM) ---
@st.cache_resource
def load_models():
    # Ensure these files exist in your 'models/' directory
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    rf = joblib.load("models/random_forest.pkl")
    lstm = load_model("models/lstm_model.keras")
    return scaler_X, scaler_y, rf, lstm

scaler_X, scaler_y, rf_model, lstm_model = load_models()

# --- HELPER FUNCTIONS ---

# 1. Get Live Data
def get_live_values():
    ids = ','.join(str(FEED_IDS[f]) for f in PARAMS)
    url = f"{EMONCMS_URL}/feed/fetch.json?ids={ids}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        values = response.json()
        return dict(zip(PARAMS, values))
    except:
        return {f: 0.0 for f in PARAMS}

# 2. Get Raw History Data
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

# 3. Get Full Historical DataFrame
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

# 4. RF Prediction (Instant)
def predict_with_rf(features_dict, scaler_X, scaler_y):
    X = np.array([[features_dict[f] for f in FEATURES]])
    X_scaled = scaler_X.transform(X)
    y_scaled = rf_model.predict(X_scaled)
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
    return max(0, y)

# 5. LSTM Forecast (Future Sequence)
def forecast_next_period_lstm(df_past, num_steps, interval_sec, start_time_future, scaler_X, scaler_y):
    if len(df_past) < TIMESTEPS:
        return pd.DataFrame()
    
    last_row = df_past.iloc[-1]
    voltage_const = last_row['voltage']
    pf_const = last_row['pf']
    frequency_const = last_row['frequency']
    current_last = last_row['current']
    energy_last = last_row['energy_kwh']
    
    recent_features = df_past[FEATURES].tail(TIMESTEPS).values
    predictions = []
    current_time = start_time_future
    
    for _ in range(num_steps):
        X_scaled = scaler_X.transform(recent_features)
        X_lstm = X_scaled.reshape(1, TIMESTEPS, len(FEATURES))
        y_scaled = lstm_model.predict(X_lstm, verbose=0)
        pred_power = max(0, scaler_y.inverse_transform(y_scaled)[0][0])
        
        interval_hours = interval_sec / 3600.0
        new_current = pred_power / (voltage_const * pf_const) if voltage_const * pf_const > 0 else current_last
        new_energy = energy_last + (pred_power * interval_hours / 1000.0)
        new_row = np.array([voltage_const, new_current, new_energy, pf_const, frequency_const])
        
        recent_features = np.vstack((recent_features[1:], new_row.reshape(1, -1)))
        predictions.append((current_time, pred_power))
        current_time += timedelta(seconds=interval_sec)
    
    df_pred = pd.DataFrame(predictions, columns=['time', 'predicted_power'])
    df_pred.set_index('time', inplace=True)
    return df_pred

# 6. Monthly Cost Forecast
def forecast_monthly(df_hist, duration_days, interval):
    interval_hours = interval / 3600.0
    total_kwh = (df_hist['power'] * interval_hours / 1000.0).sum()
    avg_daily_kwh = total_kwh / max(1, duration_days)
    monthly_kwh = avg_daily_kwh * 30
    monthly_peso = monthly_kwh * PESO_PER_KWH
    return monthly_kwh, monthly_peso

# --- UI START ---
st.set_page_config(page_title="Power Monitoring", page_icon="⚡", layout="wide")
st.title("⚡ Smart Home Energy Tracker")

# --- SIDEBAR: COLLAPSIBLE SECTIONS ---
st.sidebar.header("Configuration")

# 1. Date Selection
with st.sidebar.expander("📅 Date Range Settings", expanded=True):
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=1))
    end_date = st.date_input("End Date", value=date.today())
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()
    use_live_data = st.checkbox("Use Live Data", value=True)

# 2. Forecasting Settings
with st.sidebar.expander("🔮 Forecasting Setup", expanded=False):
    enable_forecast = st.checkbox("Enable Next Period Forecast", value=False)
    if enable_forecast:
        forecast_period = st.selectbox("Forecast Period", ["Daily", "Weekly", "Monthly"])
        past_start = st.date_input("Start of Past Period (Input)", value=date.today() - timedelta(days=1))
        
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
        else:
            past_end = past_start + relativedelta(months=1, days=-1)
            forecast_start = past_end + timedelta(days=1)
            forecast_end = forecast_start + relativedelta(months=1, days=-1)
            forecast_interval = 86400
            period_label = "Month"
            
        st.caption(f"Input: {past_start} to {past_end}")
        st.caption(f"Target: {forecast_start} to {forecast_end}")

# 3. View Filters
with st.sidebar.expander("👁️ View Options", expanded=False):
    show_params = st.checkbox("Show Parameters", value=True)
    show_predictions = st.checkbox("Show RF Predictions on Graph", value=False)
    show_graphs = st.checkbox("Show Historical Graphs", value=True)
    show_forecast = st.checkbox("Show Monthly Cost Est.", value=True)

# --- MANUAL INPUT (Only if Live Data is OFF) ---
if not use_live_data:
    with st.sidebar.expander("🎛️ Manual Controls", expanded=True):
        voltage = st.slider("Voltage", 0.0, 300.0, 240.0)
        current = st.slider("Current", 0.0, 10.0, 1.0)
        energy_kwh = st.slider("Energy (kWh)", 0.0, 100.0, 15.0)
        pf = st.slider("Power Factor", 0.0, 1.0, 0.8)
        frequency = st.slider("Frequency", 50.0, 60.0, 60.0)
        live_power = 0.0
        features_dict = {"voltage": voltage, "current": current, "energy_kwh": energy_kwh, "pf": pf, "frequency": frequency}
else:
    live_values = get_live_values()
    features_dict = {k: v for k, v in live_values.items() if k in FEATURES}
    live_power = live_values.get("power", 0.0)

# --- DATA LOADING ---
duration_days = (end_date - start_date).days + 1
selected_range_label = f"{start_date} to {end_date} ({duration_days} days)"
total_seconds = (datetime.combine(end_date, datetime.max.time()) - datetime.combine(start_date, datetime.min.time())).total_seconds()
interval = max(60, int(total_seconds / 100))

with st.spinner('Fetching historical data...'):
    df_hist = get_full_history_data(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), interval)

# --- INSTANT PREDICTION (RF) ---
pred_power = predict_with_rf(features_dict, scaler_X, scaler_y)

# --- MAIN DASHBOARD (Top Section) ---
if show_params:
    st.subheader("Current Status")
    cols = st.columns(3)
    for i, p in enumerate(PARAMS):
        val = live_values.get(p, 0.0) if use_live_data else features_dict.get(p, live_power if p == "power" else 0.0)
        cols[i % 3].metric(p.capitalize().replace("_kwh", " (kWh)"), f"{val:.2f}")

st.subheader("🤖 AI Load Analysis")
if pred_power > 1500:
    st.error(f"⚠️ **High Load:** Predicted {pred_power:.1f}W. Heavy appliances likely active.")
elif pred_power > 200:
    st.warning(f"⚠️ **Moderate Load:** Predicted {pred_power:.1f}W. Baseload higher than idle.")
else:
    st.success(f"✅ **Efficient:** Predicted {pred_power:.1f}W. Normal idle levels.")

# --- HISTORICAL TABS (Middle Section) ---
if not df_hist.empty and show_graphs:
    st.markdown("---")
    st.subheader(f"📊 Historical Trends ({selected_range_label})")
    
    # Pre-calculate RF predictions for graph if needed
    if show_predictions:
        X = df_hist[FEATURES].values
        if len(X) > 0:
            X_scaled = scaler_X.transform(X)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            df_hist['predicted'] = [max(0, p) for p in y_pred]

    # Tabs for Historical Data
    tab1, tab2, tab3 = st.tabs(["⚡ Power & Cost", "🔌 Electrical Params", "🔋 Energy Consumed"])

    with tab1:
        st.markdown("### Power Consumption")
        if show_predictions and 'predicted' in df_hist.columns:
            st.line_chart(df_hist[['power', 'predicted']])
        else:
            st.line_chart(df_hist[['power']])
        
        # Monthly Forecast is now strictly inside this tab
        if show_forecast:
            st.markdown("---")
            st.markdown("### 📅 Monthly Forecast Projection")
            monthly_kwh, monthly_peso = forecast_monthly(df_hist, duration_days, interval)
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Projected Monthly kWh", f"{monthly_kwh:.2f}")
            col_f2.metric("Projected Monthly Cost (PHP)", f"{monthly_peso:.2f}")
            st.caption(f"Based on current range usage. Rate: {PESO_PER_KWH} PHP/kWh.")

    with tab2:
        c1, c2 = st.columns(2)
        c1.line_chart(df_hist['voltage'])
        c1.caption("Voltage (V)")
        c1.line_chart(df_hist['pf'])
        c1.caption("Power Factor")
        c2.line_chart(df_hist['current'])
        c2.caption("Current (A)")
        c2.line_chart(df_hist['frequency'])
        c2.caption("Frequency (Hz)")

    with tab3:
        st.line_chart(df_hist['energy_kwh'])
        st.caption("Cumulative Energy (kWh)")

# --- PERIOD FORECAST SECTION (SEPARATED Bottom Section) ---
if enable_forecast:
    st.markdown("---")
    st.header(f"🔮 Intelligent Forecast: Next {period_label}")
    
    df_past = pd.DataFrame()
    df_actual_forecast = pd.DataFrame()
    
    with st.spinner('Preparing forecast data...'):
        # Fetch Input
        df_past = get_full_history_data(
            datetime.combine(past_start, datetime.min.time()),
            datetime.combine(past_end, datetime.max.time()),
            forecast_interval
        )
        
        # Check for Actual Data (for validation/comparison)
        df_actual_forecast = get_full_history_data(
            datetime.combine(forecast_start, datetime.min.time()),
            datetime.combine(forecast_end, datetime.max.time()),
            forecast_interval
        )
    
    if df_past.empty:
        st.error("❌ Not enough input data to generate a forecast. Please check the 'Forecasting Setup' dates.")
    else:
        # Determine Prediction Length (Mathematical or Data-based)
        if not df_actual_forecast.empty:
            num_steps = len(df_actual_forecast)
        else:
            start_dt = datetime.combine(forecast_start, datetime.min.time())
            end_dt = datetime.combine(forecast_end, datetime.max.time())
            num_steps = int((end_dt - start_dt).total_seconds() / forecast_interval)
        
        # Run LSTM Forecast
        start_time_future = datetime.combine(forecast_start, datetime.min.time())
        df_forecast = forecast_next_period_lstm(df_past, num_steps, forecast_interval, start_time_future, scaler_X, scaler_y)
        
        if not df_forecast.empty:
            # Layout: Graph on Left, Metrics on Right
            col_main, col_metrics = st.columns([2, 1])
            
            with col_main:
                st.subheader("Forecasted Power Usage")
                if df_actual_forecast.empty:
                    # PURE FORECAST (Future)
                    st.line_chart(df_forecast)
                else:
                    # VALIDATION (Past/Present)
                    compare_df = df_actual_forecast[['power']].join(df_forecast, how='inner').dropna()
                    compare_df.columns = ['Actual', 'Predicted']
                    if not compare_df.empty:
                        st.line_chart(compare_df)
                    else:
                        st.warning("Time alignment issue. Showing forecast only.")
                        st.line_chart(df_forecast)

            with col_metrics:
                st.subheader("Analysis")
                avg_p = df_forecast['predicted_power'].mean()
                peak_p = df_forecast['predicted_power'].max()
                st.metric("Predicted Avg Power", f"{avg_p:.1f} W")
                st.metric("Predicted Peak Power", f"{peak_p:.1f} W")
                
                if not df_actual_forecast.empty:
                    st.divider()
                    st.caption("Accuracy vs Actual Data")
                    # Calculate errors if we have comparison data
                    if not df_actual_forecast.empty:
                        compare_df = df_actual_forecast[['power']].join(df_forecast, how='inner').dropna()
                        compare_df.columns = ['Actual', 'Predicted']
                        if not compare_df.empty:
                            y_true = compare_df['Actual']
                            y_pred = compare_df['Predicted']
                            mae = np.mean(np.abs(y_true - y_pred))
                            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                            st.metric("MAE (Error)", f"{mae:.1f} W")
                            st.metric("RMSE", f"{rmse:.1f} W")