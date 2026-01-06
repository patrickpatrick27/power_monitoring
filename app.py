import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import load_model

# =========================================================
# CONFIGURATION
# =========================================================
EMONCMS_URL = "https://emoncms.org"
API_KEY = "8fd531f2a88ab44e99029a9c68f6497a"  # READ-ONLY
TIMESTEPS = 5
PESO_PER_KWH = 13

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

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_models():
    try:
        scaler_X = joblib.load("models/scaler_X.pkl")
        scaler_y = joblib.load("models/scaler_y.pkl")
        rf = joblib.load("models/random_forest.pkl")
        xgb = joblib.load("models/xgboost.pkl")
        lstm = load_model("models/lstm_model.keras")
        return scaler_X, scaler_y, rf, xgb, lstm
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None, None, None

scaler_X, scaler_y, rf_model, xgb_model, lstm_model = load_models()

# =========================================================
# UTILITIES
# =========================================================
def inverse_y(y_scaled):
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    return scaler_y.inverse_transform(y_scaled).flatten()

# =========================================================
# DATA FETCHING
# =========================================================
def get_live_values():
    try:
        ids = ",".join(str(FEED_IDS[p]) for p in PARAMS)
        url = f"{EMONCMS_URL}/feed/fetch.json?ids={ids}&apikey={API_KEY}"
        r = requests.get(url, timeout=5).json()
        return dict(zip(PARAMS, r))
    except:
        return {p: 0.0 for p in PARAMS}

def get_feed_history(fid, start, end, interval):
    try:
        start_s = int(start.timestamp())
        end_s = int(end.timestamp())
        url = (
            f"{EMONCMS_URL}/feed/data.json?"
            f"id={fid}&start={start_s}&end={end_s}"
            f"&interval={interval}&skipmissing=1&limitinterval=1"
            f"&apikey={API_KEY}"
        )
        return requests.get(url, timeout=10).json()
    except:
        return []

def get_full_history(start, end, interval):
    base = get_feed_history(FEED_IDS["power"], start, end, interval)
    if not base:
        return pd.DataFrame()

    times = [pd.to_datetime(x[0], unit="s") for x in base]
    df = pd.DataFrame(index=times)
    df["power"] = [x[1] for x in base]

    for f in FEATURES:
        data = get_feed_history(FEED_IDS[f], start, end, interval)
        if len(data) == len(base):
            df[f] = [x[1] for x in data]
        else:
            df[f] = df[f].ffill().fillna(0)

    return df.fillna(0)

def get_recent_data():
    rows = []
    for f in FEATURES:
        fid = FEED_IDS[f]
        url = f"{EMONCMS_URL}/feed/data.json?id={fid}&dp={TIMESTEPS}&apikey={API_KEY}"
        try:
            r = requests.get(url, timeout=5).json()
            rows.append([x[1] for x in r][-TIMESTEPS:])
        except:
            rows.append([0.0] * TIMESTEPS)

    df = pd.DataFrame(np.array(rows).T, columns=FEATURES)
    if len(df) < TIMESTEPS:
        last = df.iloc[-1] if not df.empty else [0]*len(FEATURES)
        while len(df) < TIMESTEPS:
            df = pd.concat([pd.DataFrame([last], columns=FEATURES), df])
    return df.astype(float)

# =========================================================
# PREDICTIONS
# =========================================================
def predict_tree(model, features):
    X = np.array([[features[f] for f in FEATURES]])
    Xs = scaler_X.transform(X)
    y = model.predict(Xs)
    return max(0, inverse_y(y)[0])

def predict_lstm_step(recent_df):
    X = scaler_X.transform(recent_df.values)
    X = X.reshape(1, TIMESTEPS, len(FEATURES))
    y = lstm_model.predict(X, verbose=0)
    return max(0, inverse_y(y)[0])

def forecast_lstm(df_past, steps, interval):
    if len(df_past) < TIMESTEPS:
        return pd.DataFrame()

    window = df_past[FEATURES].tail(TIMESTEPS).values
    t = df_past.index[-1]
    energy = df_past.iloc[-1]["energy_kwh"]
    voltage = df_past.iloc[-1]["voltage"]
    pf = df_past.iloc[-1]["pf"]

    out = []
    for _ in range(steps):
        Xs = scaler_X.transform(window)
        Xl = Xs.reshape(1, TIMESTEPS, len(FEATURES))
        y = lstm_model.predict(Xl, verbose=0)
        power = max(0, inverse_y(y)[0])

        hours = interval / 3600
        current = power / (voltage * pf) if voltage * pf > 0 else 0
        energy += power * hours / 1000

        new_row = np.array([voltage, current, energy, pf, window[-1,4]])
        window = np.vstack([window[1:], new_row])

        t += timedelta(seconds=interval)
        out.append((t, power))

    return pd.DataFrame(out, columns=["time", "predicted_power"]).set_index("time")

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config("Smart Energy Monitor", "⚡", layout="wide")
st.title("⚡ Smart Home Energy Monitoring & Forecasting")

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Prediction Model", ["Random Forest", "XGBoost", "LSTM"])
use_live = st.sidebar.checkbox("Use Live Data", True)

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=1))
end_date = st.sidebar.date_input("End Date", date.today())

interval = max(60, int(
    (datetime.combine(end_date, datetime.max.time()) -
     datetime.combine(start_date, datetime.min.time())).total_seconds() / 100
))

# =========================================================
# DATA
# =========================================================
live = get_live_values() if use_live else {}
features = {f: live.get(f, 0.0) for f in FEATURES}

df_hist = get_full_history(
    datetime.combine(start_date, datetime.min.time()),
    datetime.combine(end_date, datetime.max.time()),
    interval
)

# =========================================================
# SINGLE PREDICTION
# =========================================================
if model_choice == "Random Forest":
    pred = predict_tree(rf_model, features)
elif model_choice == "XGBoost":
    pred = predict_tree(xgb_model, features)
else:
    recent = get_recent_data()
    pred = predict_lstm_step(recent)

st.subheader("🔮 Current Power Prediction")
st.metric("Predicted Power (W)", f"{pred:.2f}")

# =========================================================
# FORECAST
# =========================================================
st.subheader("📈 Forecast (Next Period)")
if st.checkbox("Enable Forecast"):
    steps = 24
    df_forecast = forecast_lstm(df_hist, steps, interval)
    if not df_forecast.empty:
        st.line_chart(df_forecast)

# =========================================================
# HISTORY
# =========================================================
if not df_hist.empty:
    st.subheader("📊 Historical Power Usage")
    st.line_chart(df_hist["power"])

# =========================================================
# MONTHLY COST
# =========================================================
if not df_hist.empty:
    hours = interval / 3600
    kwh = (df_hist["power"] * hours / 1000).sum()
    monthly_kwh = kwh / max(1, (end_date - start_date).days) * 30
    st.subheader("💸 Estimated Monthly Cost")
    st.metric("Monthly kWh", f"{monthly_kwh:.2f}")
    st.metric("Monthly Cost (PHP)", f"{monthly_kwh * PESO_PER_KWH:.2f}")
