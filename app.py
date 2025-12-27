import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Load trained ML model and sample data
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("/Users/lakshyasmac/Desktop/Fraud detection pipeline/ieee-fraud-detection/fraud_detection_model_catboost.pkl")

@st.cache_data
def load_feature_defaults():
    """Load statistical defaults (median/mode) from training data for neutral baseline"""
    train_identity = pd.read_csv('train_identity.csv')
    train_transaction = pd.read_csv('train_transaction.csv')
    
    train = pd.merge(
        train_transaction,
        train_identity,
        on='TransactionID',
        how='left'
    )
    
    # Drop sparse columns (same as training)
    cols_to_remove_85 = [
        "D13","D14","D12","id_03","id_04","D6","id_33",
        "id_09","D8","D9","id_10","id_30","id_32","id_34",
        "id_14","V156","V161","V158","V162","V163"
    ]
    train.drop(columns=cols_to_remove_85, inplace=True, errors="ignore")
    
    # Row-level missingness features
    train["row_missing_ratio"] = train.isna().mean(axis=1)
    train["row_missing_count"] = train.isna().sum(axis=1)
    
    # ProductCD grouping
    train["ProductCD"] = train["ProductCD"].apply(
        lambda x: "W" if x == "W" else "OTHER"
    )
    
    # TransactionDT features
    train["Transaction_hour"] = (train["TransactionDT"] / 3600) % 24
    train["Transaction_day"] = (train["TransactionDT"] / 86400).astype(int)
    train["Transaction_week"] = (train["TransactionDT"] / 604800).astype(int)
    train.drop(columns=["TransactionDT"], inplace=True)
    
    # TransactionAmt log and z-score
    train["Log_TransactionAmt"] = np.log1p(train["TransactionAmt"])
    amt_mean = train["Log_TransactionAmt"].mean()
    amt_std = train["Log_TransactionAmt"].std()
    train["Log_TransactionAmt_z"] = (
        train["Log_TransactionAmt"] - amt_mean
    ) / (amt_std + 1e-9)
    train.drop(columns=["TransactionAmt", "Log_TransactionAmt"], inplace=True)
    
    # Get feature columns (after engineering)
    X = train.drop(columns=["isFraud", "TransactionID"]).copy()
    
    # Create a default row with median/mode values
    defaults = {}
    
    # Numeric features: use median (skip NaN)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = X[col].median()
        defaults[col] = median_val if pd.notna(median_val) else 0
    
    # Categorical features: use mode (most common)
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = X[col].mode()
        defaults[col] = mode_val[0] if len(mode_val) > 0 else "missing"
    
    # Store amt_mean and amt_std for later use
    defaults['_amt_mean'] = amt_mean
    defaults['_amt_std'] = amt_std
    
    # Create a DataFrame with one row
    default_df = pd.DataFrame([defaults])
    
    return default_df, amt_mean, amt_std

# Load model and defaults
model = load_model()
default_row, amt_mean, amt_std = load_feature_defaults()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

st.title("💳 Fraud Detection – Transaction Simulator")
st.caption(
    "This demo simulates a **real-world card transaction**. "
    "Only business-level fields are editable. "
    "All technical features are auto-derived internally."
)

st.divider()

# =============================
# 🧾 Transaction Details
# =============================
st.subheader("🧾 Transaction Details")

transaction_amount = st.number_input(
    "Transaction Amount (USD)",
    min_value=0.0,
    value=249.99,
    step=1.0,
    help="Total amount charged to the customer"
)

product_cd = st.selectbox(
    "Purchase Category",
    ["W", "C", "R", "H", "S"],
    format_func=lambda x: {
        "W": "W – Online Retail (Most common)",
        "C": "C – Card Present",
        "R": "R – Recurring Payment",
        "H": "H – Home & Utility",
        "S": "S – Subscription"
    }[x],
    help="High-level transaction category used by the payment system"
)

st.divider()

# =============================
# 💳 Payment Method
# =============================
st.subheader("💳 Payment Method")

col1, col2 = st.columns(2)

with col1:
    card_network = st.selectbox(
        "Card Network",
        ["Visa", "Mastercard", "Amex", "Discover"],
        help="Issuing network of the payment card"
    )

with col2:
    card_type = st.selectbox(
        "Card Type",
        ["Credit", "Debit"],
        help="Type of card used for payment"
    )

st.divider()

# =============================
# 🕒 Time of Transaction
# =============================
st.subheader("🕒 Transaction Time")

col3, col4 = st.columns(2)

with col3:
    transaction_date = st.date_input(
        "Transaction Date",
        help="Date when the transaction occurred"
    )

with col4:
    transaction_time_input = st.time_input(
        "Transaction Time",
        help="Local time of transaction"
    )

transaction_time = datetime.combine(transaction_date, transaction_time_input)

st.divider()

# =============================
# ⚙️ Advanced / Hidden Details
# =============================
with st.expander("⚙️ Advanced Details (Auto-filled)", expanded=False):
    st.markdown(
        """
        The following fields are **not entered manually** in real systems.
        They are derived automatically from historical data or backend logs:
        
        - Card identifiers (card1, card2, card3, card5)
        - Device & account behavior signals
        - Velocity & frequency features
        - Aggregated fraud history
        
        These are populated using a **reference transaction template** 
        to ensure the model receives all required features.
        """
    )

# =============================
# Prediction Button
# =============================
st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button(
    "🔍 Analyze Fraud Risk",
    type="primary",
    use_container_width=True
)

# =============================
# Prediction Logic
# =============================
if predict_clicked:
    try:
        # Feature engineering function
        def engineer_features(user_input):
            """Engineer features from user input"""
            return {
                "TransactionDT": int(user_input["timestamp"].timestamp()),
                "card4": user_input["card_network"].lower(),
                "card6": user_input["card_type"].lower()
            }
        
        def prepare_features():
            """Create input DataFrame with all required columns using statistical defaults"""
            # Prepare user input dictionary
            user_input = {
                "timestamp": transaction_time,
                "card_network": card_network,
                "card_type": card_type
            }
            
            # Engineer features from user input
            engineered = engineer_features(user_input)
            
            # Start with default row (median/mode values for neutral baseline)
            input_df = default_row.copy()
            
            # Remove internal metadata columns
            if '_amt_mean' in input_df.columns:
                input_df = input_df.drop(columns=['_amt_mean', '_amt_std'])
            
            # Calculate time features from TransactionDT
            transaction_dt = engineered["TransactionDT"]
            input_df["Transaction_hour"] = (transaction_dt / 3600) % 24
            input_df["Transaction_day"] = int(transaction_dt / 86400)
            input_df["Transaction_week"] = int(transaction_dt / 604800)
            
            # TransactionAmt processing (match training pipeline)
            log_amt = np.log1p(transaction_amount)
            input_df["Log_TransactionAmt_z"] = (log_amt - amt_mean) / (amt_std + 1e-9)
            
            # Update product code (transform to W or OTHER)
            product_cd_transformed = "W" if product_cd == "W" else "OTHER"
            input_df["ProductCD"] = product_cd_transformed
            
            # Update card features from user input
            input_df["card4"] = engineered["card4"]
            input_df["card6"] = engineered["card6"]
            
            # Recalculate row missingness (after updates)
            input_df["row_missing_ratio"] = input_df.isna().mean(axis=1)
            input_df["row_missing_count"] = input_df.isna().sum(axis=1)
            
            # Ensure column order matches training
            expected_cols = [c for c in default_row.columns if not c.startswith('_')]
            input_df = input_df[expected_cols]
            
            return input_df
        
        # Prepare features and make prediction
        input_df = prepare_features()
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.divider()
        
        # =============================
        # Display Results
        # =============================
        st.subheader("📊 Fraud Risk Analysis")
        
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED**")
            st.metric("Fraud Risk Score", f"{probability:.2%}", delta=None)
            st.warning("⚠️ This transaction has been flagged as potentially fraudulent. Please review manually.")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**")
            st.metric("Fraud Risk Score", f"{probability:.2%}", delta=None)
            st.info("ℹ️ This transaction appears to be legitimate based on the model analysis.")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71' if prediction == 0 else '#e74c3c']
        ax.bar(["Legitimate", "Fraud"], [1 - probability, probability], 
               color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title("Fraud Risk Distribution", fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add probability labels on bars
        ax.text(0, 1 - probability + 0.02, f"{(1-probability):.1%}", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(1, probability + 0.02, f"{probability:.1%}", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        st.pyplot(fig)
        
        # Show input summary
        with st.expander("🔎 View Transaction Summary", expanded=False):
            summary = {
                "Transaction Amount": f"${transaction_amount:,.2f}",
                "Purchase Category": product_cd,
                "Card Network": card_network,
                "Card Type": card_type,
                "Transaction Date": transaction_date.strftime("%Y-%m-%d"),
                "Transaction Time": transaction_time_input.strftime("%H:%M:%S"),
                "Transaction Hour": f"{input_df['Transaction_hour'].iloc[0]:.1f}",
                "Fraud Risk": f"{probability:.2%}"
            }
            st.json(summary)
            
    except Exception as e:
        st.error(f"❌ Error analyzing transaction: {str(e)}")
        with st.expander("🔍 Technical Details"):
            st.exception(e)
