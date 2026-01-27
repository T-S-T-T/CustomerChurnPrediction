"""
streamlit_app.py

Run:
    streamlit run streamlit_app.py

This Streamlit app:
- Loads artifacts from ./artifacts/
- Shows dataset sample and feature distributions
- Allows uploading a CSV with same schema (customer_id, usage_type, Day_1..Day_90, optional churn)
- Runs predictions and shows probabilities and a simple dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACT_DIR = "./artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'xgb_model.joblib')
ENCODER_PATH = os.path.join(ARTIFACT_DIR, 'target_encoder.joblib')
SCALER_PATH = os.path.join(ARTIFACT_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(ARTIFACT_DIR, 'model_features.joblib')
ROC_PATH = os.path.join(ARTIFACT_DIR, 'roc_curve.png')
FI_PATH = os.path.join(ARTIFACT_DIR, 'feature_importance.png')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Customer Churn Prediction — Demo")
st.markdown("Upload a CSV with the same schema as the training data (customer_id, usage_type, Day_1..Day_90, churn optional).")

# Load artifacts
if not os.path.exists(MODEL_PATH):
    st.error("Model artifacts not found. Please run `train_model.py` first to create artifacts in ./artifacts/")
    st.stop()

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(FEATURES_PATH)

@st.cache_data
def load_sample_data():
    # Try to load original data if available
    sample_path = "/data/main.csv"
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path).head(200)
    return None

sample_df = load_sample_data()

col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
    else:
        if sample_df is not None:
            st.info("No file uploaded — using sample rows from /data/main.csv")
            df_input = sample_df.copy()
        else:
            st.warning("No sample data available. Upload a CSV to proceed.")
            st.stop()

    st.subheader("Preview of input data")
    st.dataframe(df_input.head(10))

with col2:
    st.subheader("Model artifacts")
    st.write("Model:", os.path.basename(MODEL_PATH))
    st.write("Encoder:", os.path.basename(ENCODER_PATH))
    st.write("Scaler:", os.path.basename(SCALER_PATH))
    if os.path.exists(ROC_PATH):
        st.image(ROC_PATH, caption="ROC Curve", use_column_width=True)
    if os.path.exists(FI_PATH):
        st.image(FI_PATH, caption="Feature importance", use_column_width=True)

# Feature engineering function (same logic as training)
def get_day_columns(df):
    return [c for c in df.columns if c.startswith('Day_')]

def feature_engineer_for_app(df):
    df_fe = df.copy()
    day_cols = get_day_columns(df_fe)
    df_fe[day_cols] = df_fe[day_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_fe['usage_total'] = df_fe[day_cols].sum(axis=1)
    df_fe['usage_mean'] = df_fe[day_cols].mean(axis=1)
    df_fe['usage_std'] = df_fe[day_cols].std(axis=1).fillna(0.0)
    df_fe['usage_min'] = df_fe[day_cols].min(axis=1)
    df_fe['usage_max'] = df_fe[day_cols].max(axis=1)
    df_fe['usage_median'] = df_fe[day_cols].median(axis=1)
    df_fe['days_nonzero'] = (df_fe[day_cols] > 0).sum(axis=1)
    last_7 = day_cols[-7:] if len(day_cols) >= 7 else day_cols
    last_30 = day_cols[-30:] if len(day_cols) >= 30 else day_cols
    df_fe['last_7_mean'] = df_fe[last_7].mean(axis=1)
    df_fe['last_30_mean'] = df_fe[last_30].mean(axis=1)
    if len(day_cols) >= 14:
        prev_7 = day_cols[-14:-7]
        df_fe['prev_7_mean'] = df_fe[prev_7].mean(axis=1)
        df_fe['pct_change_last7_prev7'] = (df_fe['last_7_mean'] - df_fe['prev_7_mean']) / (df_fe['prev_7_mean'] + 1e-9)
    else:
        df_fe['prev_7_mean'] = 0.0
        df_fe['pct_change_last7_prev7'] = 0.0

    # Trend slope
    X = np.arange(len(day_cols)).reshape(-1, 1)
    slopes = []
    for idx, row in df_fe[day_cols].iterrows():
        y = row.values.reshape(-1, 1)
        if np.allclose(y.flatten(), 0):
            slopes.append(0.0)
            continue
        lr = LinearRegression()
        try:
            lr.fit(X, y)
            slopes.append(float(lr.coef_[0]))
        except:
            slopes.append(0.0)
    df_fe['usage_trend_slope'] = slopes

    df_fe['usage_mean_to_total_ratio'] = df_fe['usage_mean'] / (df_fe['usage_total'] + 1e-9)
    df_fe['usage_std_to_mean'] = df_fe['usage_std'] / (df_fe['usage_mean'] + 1e-9)
    return df_fe

# Apply feature engineering
from sklearn.linear_model import LinearRegression
df_fe = feature_engineer_for_app(df_input)

# Prepare features for model
required_features = model_features  # list saved during training
missing = [f for f in required_features if f not in df_fe.columns]
if missing:
    st.error(f"Missing required features for model: {missing}")
    st.stop()

X_app = df_fe[required_features].copy()

# Encode usage_type with saved encoder
# encoder.transform expects a DataFrame with the column
X_app_enc = X_app.copy()
try:
    X_app_enc['usage_type'] = encoder.transform(X_app_enc[['usage_type']])['usage_type']
except Exception as e:
    st.error(f"Error encoding usage_type: {e}")
    st.stop()

# Scale numeric features
numeric_cols = [c for c in X_app_enc.columns if c != 'usage_type']
X_app_enc[numeric_cols] = scaler.transform(X_app_enc[numeric_cols])

# Predict
probs = model.predict_proba(X_app_enc)[:, 1]
preds = (probs >= 0.5).astype(int)

result_df = df_input[['customer_id']].copy() if 'customer_id' in df_input.columns else pd.DataFrame({'index': df_input.index})
result_df['churn_proba'] = probs
result_df['churn_pred'] = preds

st.subheader("Predictions")
st.dataframe(result_df.head(50))

# Summary charts
st.subheader("Prediction distribution")
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.histplot(result_df['churn_proba'], bins=30, ax=ax[0])
ax[0].set_title("Churn probability distribution")
sns.countplot(x='churn_pred', data=result_df, ax=ax[1])
ax[1].set_title("Predicted classes (0=stay,1=churn)")
st.pyplot(fig)

# If true labels present, show metrics
if 'churn' in df_input.columns:
    y_true = df_input['churn'].astype(int)
    from sklearn.metrics import roc_auc_score, classification_report
    auc = roc_auc_score(y_true, probs)
    st.write(f"ROC-AUC on uploaded data: **{auc:.4f}**")
    st.text(classification_report(y_true, preds))

st.markdown("**Notes:** This demo uses aggregated features from Day_1..Day_90 and a model trained with SMOTE and XGBoost. For production, consider monitoring data drift and retraining periodically.")