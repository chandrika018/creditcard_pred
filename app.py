import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it is Fraud or Normal.")

# 30 features
feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

if st.button("Predict"):

    data = np.array(inputs).reshape(1, -1)

    # Scale input
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Normal Transaction")