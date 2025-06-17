import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# GitHub CSV URL (raw link)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/your_data.csv"

# Load data from GitHub
@st.cache_data
def load_data_from_github():
    return pd.read_csv(GITHUB_RAW_URL)

# UI
st.set_page_config(page_title="FraudWatch AI", layout="centered")
st.title("🛡️ FraudWatch AI")
st.markdown("Detect fraudulent transactions using a trained XGBoost model on the PaySim dataset.")

# Choose data source
option = st.radio("Choose data source:", ["From GitHub", "Upload CSV"])

data = None

if option == "From GitHub":
    st.info(f"Loading data from GitHub: {GITHUB_RAW_URL}")
    try:
        data = load_data_from_github()
    except:
        st.error("❌ Failed to load from GitHub. Check the URL.")
else:
    uploaded_file = st.file_uploader("📤 Upload a transaction CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

if data is not None:
    st.subheader("🔍 Input Data Preview")
    st.dataframe(data.head())

    # Check required columns
    required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    if not all(col in data.columns for col in required_cols):
        st.error("❌ Input data must contain all required columns.")
    else:
        # Feature engineering
        df = data.copy()
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

        # Ensure model expected columns exist
        for col in ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
            if col not in df.columns:
                df[col] = 0

        # Model features
        features = ['step', 'amount', 'balance_diff_orig', 'balance_diff_dest',
                    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        X_input = df[features]

        # Predict
        predictions = model.predict(X_input)
        df['Fraud_Prediction'] = predictions

        st.subheader("🧠 Prediction Results")
        st.dataframe(df[['amount', 'type_TRANSFER', 'type_CASH_OUT', 'Fraud_Prediction']].head(10))

        st.success(f"✅ {df['Fraud_Prediction'].sum()} potential frauds detected.")

        # Download button
        csv_out = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Predictions", data=csv_out, file_name="fraud_predictions.csv", mime='text/csv')
else:
    st.info("📎 Awaiting data input...")
