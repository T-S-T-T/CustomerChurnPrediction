"""
train_model.py

Usage:
    python train_model.py

This script:
- Loads /data/main.csv
- Performs data cleaning and feature engineering for Day_1..Day_90
- Encodes usage_type
- Handles class imbalance with SMOTE
- Trains an XGBoost classifier with cross-validated hyperparameter search
- Evaluates using ROC-AUC and saves model artifacts (model, scaler, encoder)
- Saves artifacts to ./artifacts/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

# Paths
DATA_PATH = "/data/main.csv"
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    # Ensure expected columns exist
    # Drop duplicates by customer_id
    if 'customer_id' in df.columns:
        df = df.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    # If churn label exists, ensure it's binary 0/1; if not present, create a synthetic label placeholder
    if 'churn' not in df.columns:
        # If dataset doesn't have churn, we cannot train; raise error
        raise ValueError("Dataset must contain a 'churn' column with 0/1 labels.")
    # Convert churn to int
    df['churn'] = df['churn'].astype(int)
    return df

def get_day_columns(df):
    # Day_1 .. Day_90 expected
    day_cols = [c for c in df.columns if c.startswith('Day_')]
    # Sort by day number
    def day_key(c):
        try:
            return int(c.split('_')[1])
        except:
            return 999
    day_cols = sorted(day_cols, key=day_key)
    return day_cols

def feature_engineering(df):
    df_fe = df.copy()
    day_cols = get_day_columns(df_fe)
    # Convert day columns to numeric
    df_fe[day_cols] = df_fe[day_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Basic aggregated features
    df_fe['usage_total'] = df_fe[day_cols].sum(axis=1)
    df_fe['usage_mean'] = df_fe[day_cols].mean(axis=1)
    df_fe['usage_std'] = df_fe[day_cols].std(axis=1).fillna(0.0)
    df_fe['usage_min'] = df_fe[day_cols].min(axis=1)
    df_fe['usage_max'] = df_fe[day_cols].max(axis=1)
    df_fe['usage_median'] = df_fe[day_cols].median(axis=1)
    df_fe['days_nonzero'] = (df_fe[day_cols] > 0).sum(axis=1)

    # Recent activity features
    last_7 = day_cols[-7:] if len(day_cols) >= 7 else day_cols
    last_30 = day_cols[-30:] if len(day_cols) >= 30 else day_cols
    df_fe['last_7_mean'] = df_fe[last_7].mean(axis=1)
    df_fe['last_30_mean'] = df_fe[last_30].mean(axis=1)
    # Percent change between last 7 and previous 7 (if available)
    if len(day_cols) >= 14:
        prev_7 = day_cols[-14:-7]
        df_fe['prev_7_mean'] = df_fe[prev_7].mean(axis=1)
        df_fe['pct_change_last7_prev7'] = (df_fe['last_7_mean'] - df_fe['prev_7_mean']) / (df_fe['prev_7_mean'] + 1e-9)
    else:
        df_fe['prev_7_mean'] = 0.0
        df_fe['pct_change_last7_prev7'] = 0.0

    # Trend (slope) across days using linear regression per row
    X = np.arange(len(day_cols)).reshape(-1, 1)
    slopes = []
    for idx, row in df_fe[day_cols].iterrows():
        y = row.values.reshape(-1, 1)
        # If all zeros, slope = 0
        if np.allclose(y.flatten(), 0):
            slopes.append(0.0)
            continue
        lr = LinearRegression()
        try:
            lr.fit(X, y)
            slopes.append(float(lr.coef_[0]))
        except Exception:
            slopes.append(0.0)
    df_fe['usage_trend_slope'] = slopes

    # Relative features
    df_fe['usage_mean_to_total_ratio'] = df_fe['usage_mean'] / (df_fe['usage_total'] + 1e-9)
    df_fe['usage_std_to_mean'] = df_fe['usage_std'] / (df_fe['usage_mean'] + 1e-9)

    # Encode usage_type as categorical target encoding (safer for high-cardinality)
    # We'll do encoding later in pipeline; keep usage_type column
    # Drop raw day columns if you want a compact model; but keep them for potential feature importance
    # For modeling, we'll use aggregated features + usage_type
    return df_fe

def prepare_features_targets(df_fe):
    # Features to use
    features = [
        'usage_total', 'usage_mean', 'usage_std', 'usage_min', 'usage_max',
        'usage_median', 'days_nonzero', 'last_7_mean', 'last_30_mean',
        'prev_7_mean', 'pct_change_last7_prev7', 'usage_trend_slope',
        'usage_mean_to_total_ratio', 'usage_std_to_mean', 'usage_type'
    ]
    # Ensure features exist
    features = [f for f in features if f in df_fe.columns]
    X = df_fe[features].copy()
    y = df_fe['churn'].copy()
    return X, y

def build_pipeline():
    # Encoder for usage_type: target encoder (category_encoders)
    encoder = ce.TargetEncoder(cols=['usage_type'], smoothing=0.3)
    scaler = StandardScaler()
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=4,
        random_state=42
    )
    # We'll create a simple pipeline: encoder -> scaler -> classifier
    # Note: TargetEncoder expects y during fit; we'll handle it in GridSearchCV by using a wrapper pipeline
    return encoder, scaler, xgb

def train_and_evaluate(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build encoder/scaler/model
    encoder, scaler, xgb = build_pipeline()

    # Apply target encoding manually because scikit-learn pipeline + target encoder needs y during fit
    # Fit encoder on training data
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    encoder.fit(X_train_enc[['usage_type']], y_train)
    X_train_enc['usage_type'] = encoder.transform(X_train_enc[['usage_type']])['usage_type']
    X_test_enc['usage_type'] = encoder.transform(X_test_enc[['usage_type']])['usage_type']

    # Scale numeric features (all except usage_type which is now numeric)
    numeric_cols = [c for c in X_train_enc.columns if c != 'usage_type']
    scaler.fit(X_train_enc[numeric_cols])
    X_train_enc[numeric_cols] = scaler.transform(X_train_enc[numeric_cols])
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

    # Handle imbalance with SMOTE on training set
    sm = SMOTE(random_state=42, n_jobs=4)
    X_res, y_res = sm.fit_resample(X_train_enc, y_train)

    # Hyperparameter tuning (small grid for speed)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=4,
        verbose=1
    )
    grid.fit(X_res, y_res)

    best_model = grid.best_estimator_
    # Evaluate
    y_pred_proba = best_model.predict_proba(X_test_enc)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Best params: {grid.best_params_}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    # Save ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'roc_curve.png'))
    plt.close()

    # Classification threshold metrics (0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (XGBoost)
    try:
        importances = best_model.feature_importances_
        feat_names = X_train_enc.columns
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        fi.to_csv(os.path.join(ARTIFACT_DIR, 'feature_importance.csv'))
        plt.figure(figsize=(6,4))
        sns.barplot(x=fi.values[:20], y=fi.index[:20])
        plt.title('Top feature importances')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'feature_importance.png'))
        plt.close()
    except Exception as e:
        print("Could not compute feature importances:", e)

    # Save artifacts: model, encoder, scaler, columns
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, 'xgb_model.joblib'))
    joblib.dump(encoder, os.path.join(ARTIFACT_DIR, 'target_encoder.joblib'))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.joblib'))
    # Save training columns order
    joblib.dump(list(X_train_enc.columns), os.path.join(ARTIFACT_DIR, 'model_features.joblib'))

    # Save a small evaluation summary
    eval_summary = {
        'best_params': grid.best_params_,
        'roc_auc': float(roc_auc),
        'test_size': len(y_test),
        'class_distribution_train': dict(pd.Series(y_res).value_counts().to_dict()),
        'class_distribution_test': dict(pd.Series(y_test).value_counts().to_dict())
    }
    joblib.dump(eval_summary, os.path.join(ARTIFACT_DIR, 'eval_summary.joblib'))
    print("Artifacts saved to", ARTIFACT_DIR)
    return best_model, encoder, scaler, X_train_enc.columns

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Basic cleaning...")
    df = basic_cleaning(df)
    print("Feature engineering...")
    df_fe = feature_engineering(df)
    print("Preparing features and target...")
    X, y = prepare_features_targets(df_fe)
    print("Training and evaluating model...")
    train_and_evaluate(X, y)
    print("Done.")

if __name__ == "__main__":
    main()