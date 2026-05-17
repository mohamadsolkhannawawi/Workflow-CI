"""
Modelling - MLProject Entry Point
Author: Mohamad Solkhan Nawawi

PERBAIKAN:
- Hapus dagshub.init() — tracking URI di-set via MLFLOW_TRACKING_URI di ci.yml
  sebelum `mlflow run .` dijalankan. dagshub.init() menimpa URI sehingga
  run ID dari MLflow Projects tidak dikenal di DagsHub (RESOURCE_DOES_NOT_EXIST).
- Tidak perlu mlflow.start_run() — MLflow Projects sudah membuat run aktif.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
import json

# ========================
# ARGUMENT PARSER
# ========================
parser = argparse.ArgumentParser()
parser.add_argument('--train_data',        type=str, default='insurance_preprocessing/train.csv')
parser.add_argument('--test_data',         type=str, default='insurance_preprocessing/test.csv')
parser.add_argument('--n_estimators',      type=int, default=200)
parser.add_argument('--max_depth',         type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf',  type=int, default=1)
parser.add_argument('--random_state',      type=int, default=42)
args = parser.parse_args()

# ========================
# KONFIGURASI MLFLOW
# Tracking URI sudah di-set via env var MLFLOW_TRACKING_URI di ci.yml.
# Tidak perlu dagshub.init() di sini.
# ========================
mlflow.set_experiment("Insurance-CI-Pipeline")

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
print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")

# ========================
# TRAINING
# MLflow Projects sudah membuat run aktif — langsung log ke run tersebut.
# ========================
mlflow.log_param("n_estimators",      args.n_estimators)
mlflow.log_param("max_depth",         args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)
mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
mlflow.log_param("random_state",      args.random_state)

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

mlflow.log_metric("mae",      mae)
mlflow.log_metric("mse",      mse)
mlflow.log_metric("rmse",     rmse)
mlflow.log_metric("r2_score", r2)
mlflow.log_metric("mape",     mape)

print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[indices], color='steelblue')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=100)
plt.close()

# Model summary JSON
summary = {
    "mae": round(mae, 4), "mse": round(mse, 4),
    "rmse": round(rmse, 4), "r2": round(r2, 4), "mape": round(mape, 4)
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# Log model & artefak
mlflow.sklearn.log_model(model, artifact_path="model")
mlflow.log_artifact("feature_importance.png")
mlflow.log_artifact("actual_vs_predicted.png")
mlflow.log_artifact("model_summary.json")

# Simpan run_id untuk step Build Docker di ci.yml
run_id = mlflow.active_run().info.run_id
print(f"[INFO] Run ID: {run_id}")
with open("latest_run_id.txt", "w") as f:
    f.write(run_id)

print("[INFO] Training selesai!")