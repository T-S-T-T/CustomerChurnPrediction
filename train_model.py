"""
train_model.py

- Loads data/main.csv (auto-detects common locations)
- Loads optional train/train_cxid.csv and test/test_cxid.csv label files
- Merges labels, falls back to deriving churn from Day_ columns if needed
- Feature engineering on Day_1..Day_N
- Handles imbalance: uses SMOTE if imbalanced-learn is available, otherwise uses sklearn resample upsampling
- Trains XGBoost with a small grid search (ROC-AUC)
- Evaluates on provided test label set if available, otherwise on holdout split
- Saves artifacts to ./artifacts/
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (relative to script)
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
CANDIDATE_MAIN_PATHS = [
    ROOT / "data" / "main.csv",
    ROOT / "main.csv",
    Path.cwd() / "data" / "main.csv",
    Path.cwd() / "main.csv",
]
TRAIN_LABEL_PATH = ROOT / "train" / "train_cxid.csv"
TEST_LABEL_PATH = ROOT / "test" / "test_cxid.csv"

def find_main_csv():
    for p in CANDIDATE_MAIN_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find main.csv in expected locations. Put main.csv in data/ or project root.")

def load_main_and_labels():
    main_path = find_main_csv()
    print(f"Loading main data from: {main_path}")
    df = pd.read_csv(main_path)

    train_lbl = None
    test_lbl = None
    if TRAIN_LABEL_PATH.exists():
        train_lbl = pd.read_csv(TRAIN_LABEL_PATH)
        print(f"Loaded train labels: {TRAIN_LABEL_PATH} ({len(train_lbl)} rows)")
    else:
        print("No train label file found at", TRAIN_LABEL_PATH)

    if TEST_LABEL_PATH.exists():
        test_lbl = pd.read_csv(TEST_LABEL_PATH)
        print(f"Loaded test labels: {TEST_LABEL_PATH} ({len(test_lbl)} rows)")
    else:
        print("No test label file found at", TEST_LABEL_PATH)

    return df, train_lbl, test_lbl

def merge_labels_into_main(df, train_lbl, test_lbl):
    df = df.copy()
    if 'customer_id' not in df.columns:
        raise ValueError("main.csv must contain 'customer_id' column")

    def normalize(lbl):
        if lbl is None:
            return None
        if 'customer_id' not in lbl.columns or 'churn' not in lbl.columns:
            raise ValueError("Label files must contain 'customer_id' and 'churn' columns")
        out = lbl[['customer_id', 'churn']].copy()
        out['churn'] = out['churn'].astype(int)
        return out

    train_norm = normalize(train_lbl)
    test_norm = normalize(test_lbl)

    if train_norm is not None:
        df = df.merge(train_norm, on='customer_id', how='left', suffixes=('', '_train'))
        # merged 'churn' column will be present for train customers
    if test_norm is not None:
        # keep test churn separate as churn_test
        df = df.merge(test_norm.rename(columns={'churn': 'churn_test'}), on='customer_id', how='left')

    train_ids = set(train_norm['customer_id']) if train_norm is not None else set()
    test_ids = set(test_norm['customer_id']) if test_norm is not None else set()

    return df, train_ids, test_ids

def ensure_churn_label(df, train_ids, window_days=7):
    """
    - If train labels exist, keep churn for those rows.
    - For unlabeled rows, derive churn from last `window_days` of Day_ columns:
      no usage in last window_days => churn=1, else 0.
    - If no Day_ columns exist and no labels provided, raise error.
    """
    df = df.copy()
    day_cols = [c for c in df.columns if c.startswith('Day_')]
    if not day_cols and 'churn' not in df.columns and 'churn_test' not in df.columns:
        raise ValueError("No Day_ columns and no churn labels found. Provide labels or Day_ columns.")

    # Convert day columns to numeric if present
    if day_cols:
        df[day_cols] = df[day_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        last_n = day_cols[-window_days:] if len(day_cols) >= window_days else day_cols
        derived = (df[last_n].sum(axis=1) == 0).astype(int)
    else:
        derived = pd.Series(0, index=df.index)

    # If churn column exists (from train merge), fill missing with derived
    if 'churn' in df.columns:
        df['churn'] = df['churn'].where(df['churn'].notna(), derived).astype(int)
    else:
        df['churn'] = derived.astype(int)

    # Ensure churn_test is integer if present
    if 'churn_test' in df.columns:
        df['churn_test'] = df['churn_test'].fillna(-1).astype(int)

    return df

def get_day_columns(df):
    day_cols = [c for c in df.columns if c.startswith('Day_')]
    def key(c):
        try:
            return int(c.split('_')[1])
        except:
            return 999
    return sorted(day_cols, key=key)

def feature_engineering(df):
    df_fe = df.copy()
    day_cols = get_day_columns(df_fe)
    if day_cols:
        df_fe[day_cols] = df_fe[day_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Aggregations
    if day_cols:
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
        # If no day columns, create zeros for features to keep pipeline consistent
        for col in ['usage_total','usage_mean','usage_std','usage_min','usage_max','usage_median',
                    'days_nonzero','last_7_mean','last_30_mean','prev_7_mean','pct_change_last7_prev7']:
            df_fe[col] = 0.0

    # Trend slope
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

    # Keep usage_type as-is for encoding later
    return df_fe

def prepare_feature_matrix(df_fe):
    features = [
        'usage_total', 'usage_mean', 'usage_std', 'usage_min', 'usage_max',
        'usage_median', 'days_nonzero', 'last_7_mean', 'last_30_mean',
        'prev_7_mean', 'pct_change_last7_prev7', 'usage_trend_slope',
        'usage_mean_to_total_ratio', 'usage_std_to_mean', 'usage_type'
    ]
    features = [f for f in features if f in df_fe.columns]
    X = df_fe[features].copy()
    y = df_fe['churn'].astype(int)
    return X, y, features

def safe_target_encode(train_X, train_y, val_X, col='usage_type'):
    """
    Simple target encoding with smoothing to avoid leakage.
    Returns transformed train_X and val_X (copies).
    """
    train = train_X.copy()
    val = val_X.copy()
    # Compute global mean
    global_mean = train_y.mean()
    # Compute counts and means per category
    stats = train.groupby(col)['churn'].agg(['mean','count']).rename(columns={'mean':'cat_mean','count':'cat_count'})
    # smoothing
    smoothing = 1.0
    stats['enc'] = (stats['cat_mean'] * stats['cat_count'] + global_mean * smoothing) / (stats['cat_count'] + smoothing)
    mapping = stats['enc'].to_dict()
    train[col] = train[col].map(mapping).fillna(global_mean)
    val[col] = val[col].map(mapping).fillna(global_mean)
    return train, val

def try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE
    except Exception:
        return None

def upsample_with_resample(X, y):
    df = X.copy()
    df['__target__'] = y.values
    majority_class = df['__target__'].mode()[0]
    majority = df[df['__target__'] == majority_class]
    minority = df[df['__target__'] != majority_class]
    if len(minority) == 0:
        return X, y
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    resampled = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    y_res = resampled['__target__'].astype(int)
    X_res = resampled.drop(columns='__target__')
    return X_res, y_res

def train_and_evaluate(df_fe, train_ids, test_ids):
    # Prepare features and labels
    X_all, y_all, feature_cols = prepare_feature_matrix(df_fe)

    # If train_ids provided, use those rows for training; otherwise random split
    if train_ids:
        train_mask = df_fe['customer_id'].isin(train_ids)
        X_train_df = X_all[train_mask].reset_index(drop=True)
        y_train = y_all[train_mask].reset_index(drop=True)
        # For evaluation, if test_ids provided, use them; else use remaining labeled rows
        if test_ids:
            eval_mask = df_fe['customer_id'].isin(test_ids)
            X_eval_df = X_all[eval_mask].reset_index(drop=True)
            y_eval = y_all[eval_mask].reset_index(drop=True)
        else:
            # fallback: use rows not in train_ids and that have churn label
            eval_mask = ~df_fe['customer_id'].isin(train_ids)
            X_eval_df = X_all[eval_mask].reset_index(drop=True)
            y_eval = y_all[eval_mask].reset_index(drop=True)
    else:
        # No explicit train ids: do a stratified split
        X_train_df, X_eval_df, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    print(f"Training rows: {len(X_train_df)}, Evaluation rows: {len(X_eval_df)}")

    # Encode usage_type using target encoding (fit on training only)
    if 'usage_type' in X_train_df.columns:
        # create temporary DataFrames with churn for encoding
        train_enc = X_train_df.copy()
        train_enc['churn'] = y_train.values
        # Use simple smoothing mapping
        global_mean = y_train.mean()
        stats = train_enc.groupby('usage_type')['churn'].agg(['mean','count']).rename(columns={'mean':'cat_mean','count':'cat_count'})
        smoothing = 1.0
        stats['enc'] = (stats['cat_mean'] * stats['cat_count'] + global_mean * smoothing) / (stats['cat_count'] + smoothing)
        mapping = stats['enc'].to_dict()
        X_train_df['usage_type'] = X_train_df['usage_type'].map(mapping).fillna(global_mean)
        X_eval_df['usage_type'] = X_eval_df['usage_type'].map(mapping).fillna(global_mean)

    # Scale numeric features
    numeric_cols = [c for c in X_train_df.columns if c != 'usage_type']
    scaler = StandardScaler()
    scaler.fit(X_train_df[numeric_cols])
    X_train_df[numeric_cols] = scaler.transform(X_train_df[numeric_cols])
    X_eval_df[numeric_cols] = scaler.transform(X_eval_df[numeric_cols])

    # Handle imbalance: try SMOTE first
    SMOTE = try_import_smote()
    if SMOTE is not None:
        print("Using SMOTE for oversampling (imblearn available).")
        sm = SMOTE(random_state=42, n_jobs=4)
        X_res, y_res = sm.fit_resample(X_train_df, y_train)
    else:
        print("imblearn not available — using sklearn resample upsampling.")
        X_res, y_res = upsample_with_resample(X_train_df, y_train)

    # Train XGBoost with small grid search
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_jobs=4, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring='roc_auc', cv=cv, n_jobs=4, verbose=1)
    grid.fit(X_res, y_res)

    best_model = grid.best_estimator_
    # Evaluate on evaluation set
    y_proba = best_model.predict_proba(X_eval_df)[:, 1]
    roc_auc = roc_auc_score(y_eval, y_proba) if len(np.unique(y_eval)) > 1 else float('nan')
    print("Best params:", grid.best_params_)
    print(f"Evaluation ROC-AUC: {roc_auc:.4f}")

    # Save ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_eval, y_proba)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / 'roc_curve.png')
        plt.close()
    except Exception:
        pass

    # Classification report at 0.5
    y_pred = (y_proba >= 0.5).astype(int)
    print("Classification report (threshold=0.5):")
    print(classification_report(y_eval, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_eval, y_pred))

    # Feature importance
    try:
        fi = pd.Series(best_model.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)
        fi.to_csv(ARTIFACT_DIR / 'feature_importance.csv')
        plt.figure(figsize=(6,4))
        sns.barplot(x=fi.values[:20], y=fi.index[:20])
        plt.title('Top feature importances')
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / 'feature_importance.png')
        plt.close()
    except Exception:
        pass

    # Save artifacts
    joblib.dump(best_model, ARTIFACT_DIR / 'xgb_model.joblib')
    joblib.dump(scaler, ARTIFACT_DIR / 'scaler.joblib')
    joblib.dump(feature_cols, ARTIFACT_DIR / 'model_features.joblib')
    eval_summary = {
        'best_params': grid.best_params_,
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'train_rows': len(X_res),
        'eval_rows': len(X_eval_df),
    }
    joblib.dump(eval_summary, ARTIFACT_DIR / 'eval_summary.joblib')
    print("Saved artifacts to", ARTIFACT_DIR)

    return best_model, scaler, feature_cols

def main():
    print("Loading data and labels...")
    df_main, train_lbl, test_lbl = load_main_and_labels()
    df_merged, train_ids, test_ids = merge_labels_into_main(df_main, train_lbl, test_lbl)
    print("Ensuring churn label (deriving where missing)...")
    df_labeled = ensure_churn_label(df_merged, train_ids, window_days=7)
    print("Feature engineering...")
    df_fe = feature_engineering(df_labeled)
    print("Training and evaluation...")
    model, scaler, feature_cols = train_and_evaluate(df_fe, train_ids, test_ids)
    print("Done.")

if __name__ == "__main__":
    main()