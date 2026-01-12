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
    return scaler_X, scaler_y, rf

scaler_X, scaler_y, rf_model = load_models()

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

# --- DASHBOARD UI (All in one page) ---
st.set_page_config(page_title="Power Monitoring", page_icon="⚡", layout="wide")
st.title("⚡ Smart Home Energy Tracker")

# Sidebar for Interactivity & Filters
st.sidebar.header("Interact & Predict")
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

st.sidebar.subheader("Filter Views")
show_params = st.sidebar.checkbox("Show Parameters", value=True)
show_predictions = st.sidebar.checkbox("Show Model Predictions", value=False)
show_graphs = st.sidebar.checkbox("Show Graphs", value=True)
show_forecast = st.sidebar.checkbox("Show Monthly Forecast", value=True)

# ────────────────────────────────────────────────────────────────
#               TESTING & VALIDATION SECTION
# ────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("🧪 Testing & Validation")

test_phase = st.sidebar.radio(
    "Select Testing Phase",
    [
        "Unit Testing",
        "Integration Testing",
        "System Testing",
        "Acceptance Testing",
        "Performance Testing",
        "Usability Testing",
        "Reliability Testing",
        "Security Testing"
    ],
    index=0
)

# Simple in-memory storage for test results (session state)
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

# Helper function to record a test result
def record_test(phase, test_id, name, actual, trials, status):
    key = f"{phase}_{test_id}"
    st.session_state.test_results[key] = {
        "Test ID": test_id,
        "Test Name": name,
        "Actual Result": actual,
        "Trials and Runs": trials,
        "Status": status
    }

# ── Different forms depending on phase ───────────────────────────────

if test_phase == "Unit Testing":
    st.sidebar.subheader("Unit Testing")
    
    ut_tests = [
        ("UT-01", "ESP32 microcontroller Latency", "1 to 10 ms"),
        ("UT-02", "PZEM-004T voltage accuracy", "±1% or better"),
        ("UT-03", "SCT-013-050 current accuracy", "±1-2%"),
        ("UT-04", "Random Forest forecast", "MAE < 50W or MAPE < 8%"),
        ("UT-05", "Dashboard testing", "All widgets load < 5s")
    ]
    
    for tid, name, exp in ut_tests:
        col1, col2, col3 = st.sidebar.columns([2,2,1])
        col1.write(tid)
        col2.write(name)
        actual = col3.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")

    # Quick Random Forest forecast error test (for UT-04)
    if st.sidebar.button("Quick Forecast Accuracy Check"):
        if not df_hist.empty and len(df_hist) > 10:
            X_test = df_hist[FEATURES].values
            y_true = df_hist['power'].values
            X_scaled = scaler_X.transform(X_test)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
            
            st.sidebar.metric("MAE (last points)", f"{mae:.1f} W")
            st.sidebar.metric("MAPE", f"{mape:.1f}%")
            
            status = "Pass" if mae < 50 and mape < 8 else "Fail"
            record_test(test_phase, "UT-04", "Random Forest forecast", f"MAE={mae:.1f}, MAPE={mape:.1f}%", 1, status)

elif test_phase == "Integration Testing":
    st.sidebar.subheader("Integration Testing")
    
    it_tests = [
        ("IT-01", "Sensor to Microcontroller"),
        ("IT-02", "Microcontroller to API"),
        ("IT-03", "API to Forecast Model"),
        ("IT-04", "Backend to Dashboard"),
        ("IT-05", "End-to-End Communication Sensor")
    ]
    
    for tid, name in it_tests:
        col1, col2, col3 = st.sidebar.columns([2,2,1])
        col1.write(tid)
        col2.write(name)
        actual = col3.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")

elif test_phase == "System Testing":
    st.sidebar.subheader("System Testing")
    
    st_tests = [
        ("ST-01", "Real-time Monitoring", "Observe end-to-end functionality in real time", "All components sync and respond in <1s delay"),
        ("ST-02", "Forecast Accuracy Over Time", "Random Forest prediction for extended durations", "Model maintains acceptable error margin"),
        ("ST-03", "Multi-device Access", "Access dashboard from multiple mobile devices", "Consistent and correct data displayed"),
        ("ST-04", "Data Loss Scenario", "Test with brief network interruption", "System retries and restores data transmission")
    ]
    
    for tid, name, desc, exp in st_tests:
        col1, col2 = st.sidebar.columns([2,3])
        col1.write(tid)
        col2.write(name)
        actual = st.sidebar.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")
    
    # Interactive test for ST-01
    if st.sidebar.button("Test Real-time Sync (<1s delay)"):
        start = datetime.now()
        _ = get_live_values()  # force fetch
        delay = (datetime.now() - start).total_seconds()
        status = "Pass" if delay < 1.0 else "Fail"
        record_test(test_phase, "ST-01", "Real-time Monitoring", f"{delay:.3f}s", 1, status)
        st.sidebar.metric("Last Delay", f"{delay:.3f}s", delta=status)

    # Quick forecast accuracy for ST-02 (using RF)
    if st.sidebar.button("Quick Forecast Accuracy Check (ST-02)"):
        if not df_hist.empty and len(df_hist) > 10:
            X_test = df_hist[FEATURES].values
            y_true = df_hist['power'].values
            X_scaled = scaler_X.transform(X_test)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
            
            st.sidebar.metric("MAE", f"{mae:.1f} W")
            st.sidebar.metric("MAPE", f"{mape:.1f}%")
            
            status = "Pass" if mae < 50 and mape < 8 else "Fail"
            record_test(test_phase, "ST-02", "Forecast Accuracy Over Time", f"MAE={mae:.1f}, MAPE={mape:.1f}%", 1, status)

elif test_phase == "Acceptance Testing":
    st.sidebar.subheader("Acceptance Testing")
    
    at_tests = [
        ("AT-01", "User Experience on App", "1 to 5"),
        ("AT-02", "Energy Forecast Usefulness", "1 to 5"),
        ("AT-03", "Data Accuracy Perception", "1 to 10"),
        ("AT-04", "Response Time Expectation", "1 to 10"),
        ("AT-05", "Overall Satisfaction", "1 to 5")
    ]
    
    for tid, name, scale in at_tests:
        max_val = 10 if "10" in scale else 5
        rating = st.sidebar.slider(name, 1, max_val, max_val // 2, key=f"rating_{tid}")
        if st.sidebar.button(f"Record {tid}"):
            record_test(test_phase, tid, name, str(rating), 1, "Pass" if rating >= (max_val // 2 + 1) else "Fail")
            st.sidebar.success(f"{tid} recorded!")

elif test_phase == "Performance Testing":
    st.sidebar.subheader("Performance Testing")
    
    pt_tests = [
        ("PT-01", "API Load Test"),
        ("PT-02", "Data Throughput"),
        ("PT-03", "Multi-User Dashboard"),
        ("PT-04", "System Uptime"),
        ("PT-05", "Real-time Forecast Response")
    ]
    
    for tid, name in pt_tests:
        col1, col2, col3 = st.sidebar.columns([2,2,1])
        col1.write(tid)
        col2.write(name)
        actual = col3.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")

    # Interactive test for PT-05 (using RF)
    if st.sidebar.button("Test Real-time Forecast Response"):
        start = datetime.now()
        features_dict = get_live_values()
        features_dict = {k: v for k, v in features_dict.items() if k in FEATURES}
        _ = predict_with_tree(rf_model, features_dict, scaler_X, scaler_y)
        delay = (datetime.now() - start).total_seconds()
        status = "Pass" if delay < 1.0 else "Fail"
        record_test(test_phase, "PT-05", "Real-time Forecast Response", f"{delay:.3f}s", 1, status)
        st.sidebar.metric("Last Forecast Delay", f"{delay:.3f}s", delta=status)

elif test_phase == "Usability Testing":
    st.sidebar.subheader("Usability Testing")
    
    ust_tests = [
        ("UST-01", "First-Time User Test", "1 to 5"),
        ("UST-02", "Feature Clarity", "1 to 5"),
        ("UST-03", "Survey Feedback", "1 to 5")
    ]
    
    for tid, name, scale in ust_tests:
        rating = st.sidebar.slider(name, 1, 5, 3, key=f"rating_{tid}")
        if st.sidebar.button(f"Record {tid}"):
            record_test(test_phase, tid, name, str(rating), 1, "Pass" if rating >= 3 else "Fail")
            st.sidebar.success(f"{tid} recorded!")

elif test_phase == "Reliability Testing":
    st.sidebar.subheader("Reliability Testing")
    
    rt_tests = [
        ("RT-01", "24-Hour Operation"),
        ("RT-02", "Network Recovery"),
        ("RT-03", "Sensor Failure")
    ]
    
    for tid, name in rt_tests:
        col1, col2, col3 = st.sidebar.columns([2,2,1])
        col1.write(tid)
        col2.write(name)
        actual = col3.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")

elif test_phase == "Security Testing":
    st.sidebar.subheader("Security Testing")
    
    sec_tests = [
        ("SEC-01", "Encrypted Transmission"),
        ("SEC-02", "Authentication Test"),
        ("SEC-03", "SQL Injection"),
        ("SEC-04", "XSS Test")
    ]
    
    for tid, name in sec_tests:
        col1, col2, col3 = st.sidebar.columns([2,2,1])
        col1.write(tid)
        col2.write(name)
        actual = col3.text_input(f"Actual {tid}", key=f"actual_{tid}")
        
        trials = st.sidebar.slider(f"Trials {tid}", 1, 20, 5, key=f"trials_{tid}")
        status = st.sidebar.selectbox(f"Status {tid}", ["Pass", "Fail", "Pending"], key=f"status_{tid}")
        
        if st.sidebar.button(f"Save {tid}", key=f"save_{tid}"):
            record_test(test_phase, tid, name, actual, trials, status)
            st.sidebar.success(f"{tid} recorded!")

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

# --- Predictions (Single Point) ---
pred_power = predict_with_tree(rf_model, features_dict, scaler_X, scaler_y)

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
        X = df_hist[FEATURES].values
        if len(X) > 0:
            X_scaled = scaler_X.transform(X)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            df_hist['predicted'] = [max(0, p) for p in y_pred]
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

# ── Automated Tests Button ────────────────────────────────────────────────
st.subheader("Automated Testing")
if st.button("Run Automated Tests"):
    with st.spinner("Running automated tests..."):
        # ST-01: Real-time sync
        start = datetime.now()
        _ = get_live_values()
        delay = (datetime.now() - start).total_seconds()
        status = "Pass" if delay < 1.0 else "Fail"
        record_test("System Testing", "ST-01", "Real-time Monitoring", f"{delay:.3f}s", 1, status)

        # UT-04 / ST-02: Forecast accuracy (RF)
        if not df_hist.empty and len(df_hist) > 10:
            X_test = df_hist[FEATURES].values
            y_true = df_hist['power'].values
            X_scaled = scaler_X.transform(X_test)
            y_scaled = rf_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
            
            status = "Pass" if mae < 50 and mape < 8 else "Fail"
            record_test("Unit Testing", "UT-04", "Random Forest forecast", f"MAE={mae:.1f}, MAPE={mape:.1f}%", 1, status)
            record_test("System Testing", "ST-02", "Forecast Accuracy Over Time", f"MAE={mae:.1f}, MAPE={mape:.1f}%", 1, status)

        # PT-05: Forecast response time (RF)
        start = datetime.now()
        _ = predict_with_tree(rf_model, features_dict, scaler_X, scaler_y)
        delay = (datetime.now() - start).total_seconds()
        status = "Pass" if delay < 1.0 else "Fail"
        record_test("Performance Testing", "PT-05", "Real-time Forecast Response", f"{delay:.3f}s", 1, status)
    
    st.success("Automated tests completed! Check results in the selected phase.")

# ── Show Summary Table ────────────────────────────────────────────────

st.subheader(f"📋 {test_phase} Results")

if st.session_state.test_results:
    results_df = pd.DataFrame.from_dict(st.session_state.test_results, orient="index")
    
    # Filter only current phase
    if test_phase == "Unit Testing":
        phase_prefix = "UT-"
    elif test_phase == "Integration Testing":
        phase_prefix = "IT-"
    elif test_phase == "System Testing":
        phase_prefix = "ST-"
    elif test_phase == "Acceptance Testing":
        phase_prefix = "AT-"
    elif test_phase == "Performance Testing":
        phase_prefix = "PT-"
    elif test_phase == "Usability Testing":
        phase_prefix = "UST-"
    elif test_phase == "Reliability Testing":
        phase_prefix = "RT-"
    elif test_phase == "Security Testing":
        phase_prefix = "SEC-"
    
    phase_df = results_df[results_df["Test ID"].str.startswith(phase_prefix)]
    
    if not phase_df.empty:
        st.dataframe(
            phase_df[["Test ID", "Test Name", "Actual Result", "Trials and Runs", "Status"]],
            use_container_width=True
        )
        
        pass_rate = (phase_df["Status"] == "Pass").mean() * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        # Color-code status
        def color_status(val):
            color = 'green' if val == 'Pass' else 'red' if val == 'Fail' else 'orange'
            return f'background-color: {color}; color: white'
        
        st.dataframe(
            phase_df.style.applymap(color_status, subset=['Status']),
            use_container_width=True
        )

        # Download button
        csv = phase_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Phase Results as CSV",
            data=csv,
            file_name=f"{test_phase}_results.csv",
            mime='text/csv'
        )
    else:
        st.info(f"No results recorded yet for {test_phase}")
else:
    st.info("No test results recorded yet. Use the sidebar to start testing or run automated tests.")

st.caption("Note: This is a simplified in-memory test logger. Results reset on app restart. "
           "For production/paper use → export to CSV or Google Sheets.")

st.info("Tip: Use checkboxes in sidebar to filter views.")