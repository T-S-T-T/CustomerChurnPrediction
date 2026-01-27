# CustomerChurnPrediction
A Python project to predict whether a customer will leave a telecom company based on service usage and demographics

pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py


Customer Churn Prediction — Project Overview
This project predicts whether a telecom customer will churn (leave the service) based on their daily usage patterns and service type. It combines a training pipeline (train_model.py) with a deployment demo (streamlit_app.py).

How the pipeline works
1. Data sources
Main features: data/main.csv

Columns:

customer_id (unique identifier)

usage_type (e.g., usage_app_facebook_daily)

Day_1 … Day_90 (numeric usage values per day)

Labels (optional):

train/train_cxid.csv → contains customer_id and churn (0 = stayed, 1 = churned) for training customers.

test/test_cxid.csv → same format for evaluation customers.

If no label files are present, churn is derived automatically: customers with no activity in the last 7 days are marked as churned.

2. Label handling
If train/test label files exist, they are merged into the main dataset by customer_id.

Training uses only customers listed in the train file.

Evaluation uses only customers listed in the test file.

If no label files exist, churn is derived from usage inactivity.

3. Feature engineering
From the 90 daily usage columns, the script computes:

Aggregates: total, mean, std, min, max, median, number of nonzero days.

Recent activity: mean usage in last 7 and last 30 days, percent change vs. previous 7 days.

Trend: slope of usage over time (linear regression).

Ratios: mean-to-total ratio, std-to-mean ratio.

Categorical encoding: usage_type is target-encoded (mapped to numeric values based on churn rate per category).

4. Preprocessing
Scaling: numeric features standardized with StandardScaler.

Imbalance handling:

If imbalanced-learn is available, SMOTE oversampling is applied to balance churn vs. non-churn.

Otherwise, a simple upsampling strategy duplicates minority class rows.

5. Model training
Algorithm: XGBoost classifier (binary:logistic).

Hyperparameter tuning: small grid search over depth, learning rate, subsample, and number of trees.

Evaluation metric: ROC-AUC.

Outputs: ROC curve, classification report, confusion matrix, feature importance.

6. Artifacts saved
After training, the following are stored in ./artifacts/:

xgb_model.joblib → trained model

scaler.joblib → fitted scaler

model_features.joblib → list of feature names used

roc_curve.png → ROC curve plot

feature_importance.csv and feature_importance.png → feature importance

eval_summary.joblib → summary of parameters and metrics

Streamlit app (streamlit_app.py)
The app provides an interactive demo:

Upload CSV with the same schema (customer_id, usage_type, Day_1..Day_N).

Feature engineering is applied automatically.

Preprocessing: usage_type encoded, numeric features scaled with saved scaler.

Prediction: churn probability and binary prediction (threshold = 0.5).

Visualization:

Distribution of churn probabilities

Count of predicted churn vs. non-churn

ROC curve and feature importance (from training artifacts)

Evaluation: if the uploaded file includes a churn column, the app computes ROC-AUC and classification report.