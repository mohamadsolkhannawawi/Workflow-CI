"""
Modelling - MLProject Entry Point
Author: Mohamad Solkhan Nawawi
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import json
import os

# ========================
# ARGUMENT PARSER
# ========================
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='insurance_preprocessing/train.csv')
parser.add_argument('--test_data', type=str, default='insurance_preprocessing/test.csv')
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf', type=int, default=1)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# ========================
# SETUP MLFLOW
# ========================
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("insurance_prediction")

# ========================
# ARTIFACT DIR
# ========================
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print(f"[INFO] Artifact dir: {ARTIFACT_DIR}")

# ========================
# LOAD DATA
# ========================
train = pd.read_csv(args.train_data)
test  = pd.read_csv(args.test_data)

X_train = train.drop('charges', axis=1)
y_train = train['charges']
X_test  = test.drop('charges', axis=1)
y_test  = test['charges']

feature_names = list(X_train.columns)

print(f"[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Test shape: {X_test.shape}")

# ========================
# TRAINING
# ========================
with mlflow.start_run():

    # Parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("random_state", args.random_state)

    # Model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)

    print(f"[INFO] MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    # ========================
    # SAVE ARTIFACTS
    # ========================
    fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
    avp_path = os.path.join(ARTIFACT_DIR, "actual_vs_predicted.png")
    js_path = os.path.join(ARTIFACT_DIR, "model_summary.json")

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(avp_path)
    plt.close()

    # Summary JSON
    summary = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4)
    }

    with open(js_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Log ke MLflow
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(fi_path)
    mlflow.log_artifact(avp_path)
    mlflow.log_artifact(js_path)

    run_id = mlflow.active_run().info.run_id
    print(f"[INFO] Run ID: {run_id}")

print("[INFO] Training selesai!")