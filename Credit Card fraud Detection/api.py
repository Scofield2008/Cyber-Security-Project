import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")

# Streamlit page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Header
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
        }
        .sub-header {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
            color: #444;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input transaction details to detect fraud with a trained ML model</div>', unsafe_allow_html=True)

# Input form
with st.form("fraud_detection_form"):
    cols = st.columns(3)
    features = []
    for i in range(1, 29):
        with cols[i % 3]:
            val = st.number_input(f"V{i}", value=0.0, step=0.01)
            features.append(val)

    amount = st.number_input("Transaction Amount", value=0.0, step=0.01)
    features.append(amount)

    submitted = st.form_submit_button("Run Detection")

if submitted:
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][prediction]

    st.markdown("---")
    if prediction == 1:
        st.error(f" Fraudulent Transaction Detected! Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"Legitimate Transaction. Confidence: {confidence*100:.2f}%")
