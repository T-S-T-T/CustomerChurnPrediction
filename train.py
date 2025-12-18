# import os
# import joblib
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH = "data/churn.csv"   # Update if needed
# ARTIFACTS_DIR = "artifacts"
# MODEL_PATH = os.path.join(ARTIFACTS_DIR, "churn_model.joblib")
# PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "churn_pipeline.joblib")
# FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")
# REPORT_PATH = os.path.join(ARTIFACTS_DIR, "metrics.txt")

# os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# # -----------------------------
# # Load dataset
# # -----------------------------
# df = pd.read_csv(DATA_PATH)

# # -----------------------------
# # Basic cleaning and normalization
# # -----------------------------
# # Standardize target
# if "Churn" in df.columns:
#     df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
# else:
#     raise ValueError("Target column 'Churn' not found. Please adjust the code to your dataset's target column.")

# # Drop rows with missing target
# df = df.dropna(subset=["Churn"]).copy()

# # Remove obvious ID-like columns if present
# drop_candidates = [c for c in ["customerID", "CustomerID", "customer_id"] if c in df.columns]
# df = df.drop(columns=drop_candidates)

# # Example: Convert total charges string to numeric if present (common in churn datasets)
# if "TotalCharges" in df.columns:
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# # -----------------------------
# # Feature selection
# # -----------------------------
# # Separate features and target
# X = df.drop(columns=["Churn"])
# y = df["Churn"].astype(int)

# # Identify types
# numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# print(f"Numeric features: {numeric_features}")
# print(f"Categorical features: {categorical_features}")
# print(f"Class distribution: {np.bincount(y)}")

# # -----------------------------
# # Preprocessing pipelines
# # -----------------------------
# numeric_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])

# preprocessor = ColumnTransformer(transformers=[
#     ("num", numeric_transformer, numeric_features),
#     ("cat", categorical_transformer, categorical_features)
# ])

# # -----------------------------
# # Model
# # -----------------------------
# # XGBoost works well on tabular and imbalanced with scale_pos_weight; we’ll also use SMOTE during training
# xgb_clf = XGBClassifier(
#     n_estimators=300,
#     max_depth=5,
#     learning_rate=0.05,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     reg_lambda=1.0,
#     random_state=42,
#     n_jobs=-1,
#     objective="binary:logistic",
#     eval_metric="auc"
# )

# # -----------------------------
# # Imbalanced pipeline (SMOTE)
# # -----------------------------
# # SMOTE should be applied after preprocessing; use imblearn Pipeline to ensure correct order
# pipeline = ImbPipeline(steps=[
#     ("preprocess", preprocessor),
#     ("smote", SMOTE(random_state=42, k_neighbors=5)),
#     ("model", xgb_clf)
# ])

# # -----------------------------
# # Train/validation split
# # -----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # -----------------------------
# # Cross-validation ROC-AUC
# # -----------------------------
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
# print(f"CV ROC-AUC: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")

# # -----------------------------
# # Fit and evaluate
# # -----------------------------
# pipeline.fit(X_train, y_train)

# y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# y_pred = (y_pred_proba >= 0.5).astype(int)

# roc = roc_auc_score(y_test, y_pred_proba)
# print(f"Test ROC-AUC: {roc:.4f}")

# print("Classification report:")
# print(classification_report(y_test, y_pred, digits=4))

# cm = confusion_matrix(y_test, y_pred)
# print("Confusion matrix:")
# print(cm)

# # -----------------------------
# # Save artifacts
# # -----------------------------
# joblib.dump(pipeline, PIPELINE_PATH)
# joblib.dump(X.columns.tolist(), FEATURES_PATH)

# with open(REPORT_PATH, "w") as f:
#     f.write(f"CV ROC-AUC mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
#     f.write(f"Test ROC-AUC: {roc:.4f}\n")
#     f.write("\nClassification report:\n")
#     f.write(classification_report(y_test, y_pred, digits=4))
#     f.write("\nConfusion matrix:\n")
#     f.write(np.array2string(cm))

# # -----------------------------
# # Optional: Feature importance (global)
# # -----------------------------
# # Extract fitted steps
# # After preprocessing, features expand due to one-hot; we’ll recover feature names
# def get_feature_names(preprocessor, numeric_features, categorical_features):
#     num_names = numeric_features
#     cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
#     cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
#     return num_names + cat_names

# feature_names = get_feature_names(
#     pipeline.named_steps["preprocess"],
#     numeric_features,
#     categorical_features
# )

# # Using XGBoost feature importance
# booster = pipeline.named_steps["model"]
# importances = booster.feature_importances_

# fi = pd.DataFrame({"feature": feature_names, "importance": importances}) \
#        .sort_values("importance", ascending=False)

# plt.figure(figsize=(8, 10))
# sns.barplot(y="feature", x="importance", data=fi.head(25), color="#2f6f91")
# plt.title("Top 25 feature importances")
# plt.tight_layout()
# plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance_top25.png"))
# plt.close()

# print("Training complete. Artifacts saved in:", ARTIFACTS_DIR)
