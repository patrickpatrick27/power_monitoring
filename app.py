import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("This app will definitely work")
st.write("If you see this live → your GitHub + Streamlit Cloud connection is perfect")

@st.cache_resource
def get_model():
    model = RandomForestRegressor(n_estimators=5)
    X = np.random.rand(50, 5)
    y = X.sum(axis=1) * 500
    model.fit(X, y)
    return model

model = get_model()
st.success("Model loaded perfectly — no .pkl files!")

v = st.slider("Voltage", 200, 260, 230)
i = st.slider("Current", 0.0, 10.0, 2.0)
pred = model.predict([[v/250, i/10, 0.5, 0.95, 50/60]])[0]

st.metric("Predicted Power", f"{pred:.0f} W")
if pred < 600:
    st.balloons()