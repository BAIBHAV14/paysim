import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(page_title="FraudWatch AI", layout="wide")

st.title("🚨 FraudWatch AI")
st.markdown("Detect fraudulent mobile payment transactions using a trained XGBoost model.")

# Load the dataset from Google Drive (for preview/demo)
@st.cache_data
def load_data():
    file_id = "1h068GvICeTlayrHJ2MRAbdwmNBRQdgBo"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url)
    return df

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")  # Make sure this file is in the same folder

df = load_data()
model = load_model()

# Sidebar input
st.sidebar.header("📥 Enter Transaction Details")

type_options = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
transaction_type = st.sidebar.selectbox("Transaction Type", type_options)
amount = st.sidebar.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
balance_diff_orig = st.sidebar.number_input("Balance Delta (Orig.)", value=10000.0)
balance_diff_dest = st.sidebar.number_input("Balance Delta (Dest.)", value=8000.0)

# One-hot encoding for transaction type
type_dict = {f"type_{t}": 0 for t in type_options}
type_dict[f"type_{transaction_type}"] = 1

# Input for prediction
input_data = {
    'step': 1,
    'amount': amount,
    'balance_diff_orig': balance_diff_orig,
    'balance_diff_dest': balance_diff_dest,
    **type_dict
}

input_df = pd.DataFrame([input_data])

# Predict
if st.sidebar.button("🔍 Predict Fraud"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Confidence: {1 - prediction_proba:.2%})")

# Optional: Show sample data
with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head(100))
