
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("Scaler.pkl")

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<p class="title">💳 Credit Card Fraud Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Transaction Fraud Detection</p>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("📌 About Project")
st.sidebar.info(
    """
    This ML model predicts whether a credit card transaction is:

    ✅ Normal Transaction

    🚨 Fraud Transaction

    Enter transaction values and click Predict.
    """
)

# Feature Names
feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

st.subheader("📝 Enter Transaction Details")

inputs = []

# Two-column layout
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )
    else:
        value = col2.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

    inputs.append(value)

st.markdown("---")

# Predict Button
if st.button("🔍 Predict Transaction"):

    data = np.array(inputs).reshape(1, -1)

    # Scale Data
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    st.subheader("📊 Prediction Result")

    # Probability (if supported)
    try:
        prob = model.predict_proba(data_scaled)[0]

        fraud_prob = prob[1] * 100

        st.write(f"Fraud Probability: {fraud_prob:.2f}%")
        st.progress(float(fraud_prob / 100))

    except:
        pass

    # Result
    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected")
        st.balloons()

    else:
        st.success("✅ Normal Transaction")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit, Scikit-Learn, NumPy & Joblib")

