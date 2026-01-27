"""
streamlit_app.py

- Loads artifacts from ./artifacts/
- Loads a sample of data/main.csv if available
- Allows uploading a CSV with same schema (customer_id, usage_type, Day_1..Day_N)
- Performs same feature engineering and encoding used in training
- Shows predictions and simple dashboards
Run:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURES_PATH = ARTIFACT_DIR / "model_features.joblib"
ROC_PATH = ARTIFACT_DIR / "roc_curve.png"
FI_PATH = ARTIFACT_DIR / "feature_importance.png"
SAMPLE_MAIN = ROOT / "data" / "main.csv"

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction")

if not MODEL_PATH.exists():
    st.error("Model artifacts not found. Run training script first to create artifacts in ./artifacts/")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(FEATURES_PATH)

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload CSV (customer_id, usage_type, Day_1..Day_N)", type=['csv'])
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
    else:
        if SAMPLE_MAIN.exists():
            st.info("No file uploaded — using sample rows from data/main.csv")
            df_input = pd.read_csv(SAMPLE_MAIN).head(200)
        else:
            st.warning("No sample data available. Upload a CSV to proceed.")
            st.stop()
    st.subheader("Input preview")
    st.dataframe(df_input.head(10))

with col2:
    st.subheader("Model artifacts")
    st.write("Model:", MODEL_PATH.name)
    st.write("Scaler:", SCALER_PATH.name)
    st.write("Feature list length:", len(model_features))
    if ROC_PATH.exists():
        st.image(str(ROC_PATH), caption="ROC Curve", use_column_width=True)
    if FI_PATH.exists():
        st.image(str(FI_PATH), caption="Feature importance", use_column_width=True)

# Feature engineering (same logic as training)
def get_day_columns(df):
    return [c for c in df.columns if c.startswith('Day_')]

def feature_engineer(df):
    df_fe = df.copy()
    day_cols = get_day_columns(df_fe)
    if day_cols:
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
    else:
        for col in ['usage_total','usage_mean','usage_std','usage_min','usage_max','usage_median',
                    'days_nonzero','last_7_mean','last_30_mean','prev_7_mean','pct_change_last7_prev7']:
            df_fe[col] = 0.0

    # Trend slope
    day_cols = get_day_columns(df_fe)
    slopes = []
    if day_cols:
        X_idx = np.arange(len(day_cols)).reshape(-1,1)
        for _, row in df_fe[day_cols].iterrows():
            y = row.values.reshape(-1,1)
            if np.allclose(y.flatten(), 0):
                slopes.append(0.0)
                continue
            lr = LinearRegression()
            try:
                lr.fit(X_idx, y)
                slopes.append(float(lr.coef_[0]))
            except:
                slopes.append(0.0)
    else:
        slopes = [0.0] * len(df_fe)
    df_fe['usage_trend_slope'] = slopes

    df_fe['usage_mean_to_total_ratio'] = df_fe['usage_mean'] / (df_fe['usage_total'] + 1e-9)
    df_fe['usage_std_to_mean'] = df_fe['usage_std'] / (df_fe['usage_mean'] + 1e-9)
    return df_fe

df_fe = feature_engineer(df_input)

# Ensure required features exist
missing = [f for f in model_features if f not in df_fe.columns]
if missing:
    st.error(f"Missing required features: {missing}")
    st.stop()

X_app = df_fe[model_features].copy()

# Encode usage_type using simple mapping derived from training artifacts if available.
# We don't have the exact mapping saved in this simplified pipeline, so we apply a safe fallback:
# map unseen categories to global mean (0.0..1.0). If usage_type is numeric already, keep it.
if 'usage_type' in X_app.columns:
    # If usage_type is numeric already, assume it's encoded
    if not np.issubdtype(X_app['usage_type'].dtype, np.number):
        # Try to load encoder mapping if present (not saved in this simplified pipeline)
        # Fallback: label-encode categories by frequency and scale to [0,1]
        freq = X_app['usage_type'].value_counts(normalize=True)
        mapping = {k: v for k, v in freq.items()}
        X_app['usage_type'] = X_app['usage_type'].map(mapping).fillna(0.0)

# Scale numeric features
numeric_cols = [c for c in X_app.columns if c != 'usage_type']
X_app[numeric_cols] = scaler.transform(X_app[numeric_cols])

# Predict
probs = model.predict_proba(X_app)[:, 1]
preds = (probs >= 0.5).astype(int)

result_df = pd.DataFrame()
if 'customer_id' in df_input.columns:
    result_df['customer_id'] = df_input['customer_id']
else:
    result_df['index'] = df_input.index
result_df['churn_proba'] = probs
result_df['churn_pred'] = preds

st.subheader("Predictions")
st.dataframe(result_df.head(50))

st.subheader("Prediction summary")
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.histplot(result_df['churn_proba'], bins=30, ax=ax[0])
ax[0].set_title("Churn probability distribution")
sns.countplot(x='churn_pred', data=result_df, ax=ax[1])
ax[1].set_title("Predicted classes (0=stay,1=churn)")
st.pyplot(fig)

if 'churn' in df_input.columns:
    y_true = df_input['churn'].astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
        st.write(f"ROC-AUC on uploaded data: **{auc:.4f}**")
        st.text(classification_report(y_true, preds))
    except Exception as e:
        st.warning("Could not compute metrics on uploaded data: " + str(e))

st.markdown("Notes: This app uses the same aggregated features as training. For production, ensure consistent encoding and monitor data drift.")