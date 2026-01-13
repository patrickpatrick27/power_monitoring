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
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    rf = joblib.load("models/random_forest.pkl")
    lstm = load_model("models/lstm_model.keras")
    return scaler_X, scaler_y, rf, lstm

scaler_X, scaler_y, rf_model, lstm_model = load_models()

# --- HELPER FUNCTIONS ---

def get_live_values():
    ids = ','.join(str(FEED_IDS[f]) for f in PARAMS)
    url = f"{EMONCMS_URL}/feed/fetch.json?ids={ids}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        values = response.json()
        return dict(zip(PARAMS, values))
    except:
        return {f: 0.0 for f in PARAMS}

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

def predict_with_rf(features_dict, scaler_X, scaler_y):
    X = np.array([[features_dict[f] for f in FEATURES]])
    X_scaled = scaler_X.transform(X)
    y_scaled = rf_model.predict(X_scaled)
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
    return max(0, y)

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

def forecast_monthly(df_hist, duration_days, interval):
    interval_hours = interval / 3600.0
    total_kwh = (df_hist['power'] * interval_hours / 1000.0).sum()
    avg_daily_kwh = total_kwh / max(1, duration_days)
    monthly_kwh = avg_daily_kwh * 30
    monthly_peso = monthly_kwh * PESO_PER_KWH
    return monthly_kwh, monthly_peso

# --- UI START ---
st.set_page_config(page_title="Power Monitoring", page_icon="⚡", layout="wide")
st.title("⚡ Home Energy Monitoring Dashboard")

# --- SIDEBAR ---
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
with st.sidebar.expander("⚡ Forecasting Setup", expanded=False):
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
with st.sidebar.expander("⚙️ View Options", expanded=False):
    show_params = st.checkbox("Show Parameters", value=True)
    # --- CHANGED DEFAULT TO TRUE HERE ---
    show_predictions = st.checkbox("Show RF Predictions on Graph", value=True)
    show_graphs = st.checkbox("Show Historical Graphs", value=True)
    show_forecast = st.checkbox("Show Monthly Cost Est.", value=True)

# --- MANUAL INPUT ---
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

# Format dates as Words (e.g., Jan 12, 2026)
start_str = start_date.strftime("%b %d, %Y")
end_str = end_date.strftime("%b %d, %Y")
selected_range_label = f"{start_str} to {end_str}"

total_seconds = (datetime.combine(end_date, datetime.max.time()) - datetime.combine(start_date, datetime.min.time())).total_seconds()
interval = max(60, int(total_seconds / 100))

with st.spinner('Fetching historical data...'):
    df_hist = get_full_history_data(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), interval)

# --- INSTANT PREDICTION (RF) ---
pred_power = predict_with_rf(features_dict, scaler_X, scaler_y)

# --- TOP DASHBOARD ---
if show_params:
    st.subheader("Current Status")
    cols = st.columns(3)
    for i, p in enumerate(PARAMS):
        val = live_values.get(p, 0.0) if use_live_data else features_dict.get(p, live_power if p == "power" else 0.0)
        cols[i % 3].metric(p.capitalize().replace("_kwh", " (kWh)"), f"{val:.2f}")

st.subheader("Load Analysis")
if pred_power > 1500:
    st.error(f"⚠️ **High Load:** Predicted {pred_power:.1f}W. Heavy appliances likely active.")
elif pred_power > 200:
    st.warning(f"⚠️ **Moderate Load:** Predicted {pred_power:.1f}W. Baseload higher than idle.")
else:
    st.success(f"✅ **Efficient:** Predicted {pred_power:.1f}W. Normal idle levels.")

# --- MIDDLE DASHBOARD (Historical) ---
if not df_hist.empty and show_graphs:
    st.markdown("---")
    st.subheader(f"📊 Historical Trends ({selected_range_label})")
    
    # Use standard datetime index for plotting to let Streamlit handle formatting
    df_chart = df_hist.copy()
    
    if show_predictions:
        X = df_hist[FEATURES].values
        if len(X) > 0:
            X_scaled = scaler_X.transform(X)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            df_chart['predicted'] = [max(0, p) for p in y_pred]

    tab1, tab2, tab3 = st.tabs(["⚡ Power & Cost", "🔌 Electrical Params", "🔋 Energy Consumed"])

    with tab1:
        st.markdown("### Power Consumption")
        if show_predictions and 'predicted' in df_chart.columns:
            st.line_chart(df_chart[['power', 'predicted']])
        else:
            st.line_chart(df_chart[['power']])
        
        if show_forecast:
            st.markdown("---")
            # --- MODIFIED LAYOUT FOR MONTHLY FORECAST HEADER ---
            fc_col1, fc_col2 = st.columns([3, 1])
            with fc_col1:
                st.markdown("### 📅 Monthly Forecast Projection")
            with fc_col2:
                st.markdown(f"**Selected Range:** {duration_days} days")
                
            monthly_kwh, monthly_peso = forecast_monthly(df_hist, duration_days, interval)
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Projected Monthly kWh", f"{monthly_kwh:.2f}")
            col_f2.metric("Projected Monthly Cost (PHP)", f"{monthly_peso:.2f}")
            st.caption(f"Based on current range usage. Rate: {PESO_PER_KWH} PHP/kWh.")

    with tab2:
        c1, c2 = st.columns(2)
        c1.line_chart(df_chart['voltage'])
        c1.caption("Voltage (V)")
        c1.line_chart(df_chart['pf'])
        c1.caption("Power Factor")
        c2.line_chart(df_chart['current'])
        c2.caption("Current (A)")
        c2.line_chart(df_chart['frequency'])
        c2.caption("Frequency (Hz)")

    with tab3:
        st.line_chart(df_chart['energy_kwh'])
        st.caption("Cumulative Energy (kWh)")

# --- BOTTOM DASHBOARD (Forecast) ---
if enable_forecast:
    st.markdown("---")
    st.header(f"⚡ Power Forecast: Next {period_label}")
    
    df_past = pd.DataFrame()
    df_actual_forecast = pd.DataFrame()
    
    with st.spinner('Preparing forecast data...'):
        # Fetch Input Data
        df_past = get_full_history_data(
            datetime.combine(past_start, datetime.min.time()),
            datetime.combine(past_end, datetime.max.time()),
            forecast_interval
        )
        # Fetch Target Data (Validation) if it exists
        df_actual_forecast = get_full_history_data(
            datetime.combine(forecast_start, datetime.min.time()),
            datetime.combine(forecast_end, datetime.max.time()),
            forecast_interval
        )
    
    if df_past.empty:
        st.error("❌ Not enough input data to generate a forecast. Please check the 'Forecasting Setup' dates.")
    else:
        # Determine how many steps to predict
        if not df_actual_forecast.empty:
            num_steps = len(df_actual_forecast)
        else:
            # Mathematical calculation for future dates
            start_dt = datetime.combine(forecast_start, datetime.min.time())
            end_dt = datetime.combine(forecast_end, datetime.max.time())
            num_steps = int((end_dt - start_dt).total_seconds() / forecast_interval)
        
        # Run Forecast
        start_time_future = datetime.combine(forecast_start, datetime.min.time())
        df_forecast = forecast_next_period_lstm(df_past, num_steps, forecast_interval, start_time_future, scaler_X, scaler_y)
        
        if not df_forecast.empty:
            # Layout: Left = Graph, Right = Analysis
            col_main, col_metrics = st.columns([2, 1])
            
            # Use standard datetime index
            df_forecast_chart = df_forecast.copy()
            
            # --- VALIDATION MODE (Actual Data Exists) ---
            if not df_actual_forecast.empty:
                compare_df = df_actual_forecast[['power']].join(df_forecast, how='inner').dropna()
                compare_df.columns = ['Actual Power', 'Predicted Power']
                
                if not compare_df.empty:
                    y_true = compare_df['Actual Power']
                    y_pred = compare_df['Predicted Power']
                    
                    # --- CALCULATE 5 METRICS ---
                    mae = np.mean(np.abs(y_true - y_pred))
                    mse = np.mean((y_true - y_pred)**2)
                    rmse = np.sqrt(mse)
                    
                    # MAPE
                    non_zero_mask = y_true != 0
                    if np.sum(non_zero_mask) > 0:
                        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    else:
                        mape = 0.0
                    
                    # R-Squared (R2)
                    ss_res = np.sum((y_true - y_pred)**2)
                    ss_tot = np.sum((y_true - np.mean(y_true))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

                    with col_main:
                        st.subheader("Forecast vs Actual")
                        st.line_chart(compare_df)
                        
                        # Detailed Validation Table
                        with st.expander("📊 View Detailed Comparison Table", expanded=True):
                            compare_df['Error (W)'] = compare_df['Actual Power'] - compare_df['Predicted Power']
                            compare_df['% Error'] = (compare_df['Error (W)'] / compare_df['Actual Power']) * 100
                            st.dataframe(compare_df.style.format("{:.2f}"))

                    with col_metrics:
                        st.subheader("Model Accuracy")
                        st.metric("Mean Abs. Error (MAE)", f"{mae:.2f} W", help="Average absolute difference.")
                        st.metric("Root Mean Sq. Error (RMSE)", f"{rmse:.2f} W", help="Penalizes larger errors.")
                        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}", help="Variance of the error.")
                        st.metric("Mean Abs. % Error (MAPE)", f"{mape:.2f} %", help="Percentage deviation.")
                        st.metric("R² Score", f"{r2:.4f}", help="Goodness of fit (1.0 is perfect).")
                        
                        st.divider()
                        if mape < 10:
                            st.success("High Accuracy")
                        elif mape < 20:
                            st.info("Moderate Accuracy")
                        else:
                            st.warning("Low Accuracy")

                else:
                    st.warning("Timestamps do not align. Showing forecast only.")
                    st.line_chart(df_forecast_chart)

            # --- PURE FORECAST MODE (No Actual Data) ---
            else:
                with col_main:
                    st.subheader(f"Future Forecast: {forecast_start} to {forecast_end}")
                    st.line_chart(df_forecast_chart)
                    
                    # Detailed Forecast Table
                    with st.expander("📋 View Forecast Data Table", expanded=True):
                        st.dataframe(df_forecast.style.format("{:.2f}"))

                with col_metrics:
                    st.subheader("Predicted Stats")
                    avg_p = df_forecast['predicted_power'].mean()
                    peak_p = df_forecast['predicted_power'].max()
                    min_p = df_forecast['predicted_power'].min()
                    
                    # Est Energy for this period
                    hours = (len(df_forecast) * forecast_interval) / 3600.0
                    total_energy = (avg_p * hours) / 1000.0  # kWh
                    est_cost = total_energy * PESO_PER_KWH

                    st.metric("Predicted Avg Power", f"{avg_p:.2f} W")
                    st.metric("Predicted Peak Power", f"{peak_p:.2f} W")
                    st.metric("Predicted Min Power", f"{min_p:.2f} W")
                    st.divider()
                    st.metric("Est. Energy (Period)", f"{total_energy:.2f} kWh")
                    st.metric("Est. Cost (Period)", f"₱ {est_cost:.2f}")
                    st.info("⚠️ Pure prediction mode (No actual data for validation).")