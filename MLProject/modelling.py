import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import argparse
import os

# ========================
# ARGUMENT
# ========================
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='insurance_preprocessing/train.csv')
parser.add_argument('--test_data', type=str, default='insurance_preprocessing/test.csv')
args = parser.parse_args()

print("=== DEBUG PATH ===")
print("Train path:", args.train_data)
print("Test path:", args.test_data)

# ========================
# VALIDASI FILE
# ========================
if not os.path.exists(args.train_data):
    raise FileNotFoundError(f"Train file not found: {args.train_data}")

if not os.path.exists(args.test_data):
    raise FileNotFoundError(f"Test file not found: {args.test_data}")

# ========================
# MLFLOW SETUP
# ========================
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("insurance_prediction")

# ========================
# LOAD DATA
# ========================
train = pd.read_csv(args.train_data)
test = pd.read_csv(args.test_data)

print("Columns:", train.columns)

if 'charges' not in train.columns:
    raise ValueError("Column 'charges' not found in dataset")

# ========================
# SPLIT DATA
# ========================
X_train = train.drop('charges', axis=1)
y_train = train['charges']
X_test = test.drop('charges', axis=1)
y_test = test['charges']

# ========================
# TRAINING
# ========================
with mlflow.start_run() as run:

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    mlflow.log_metric("mae", mae)

    # ========================
    # WAJIB (REMOTE LOG)
    # ========================
    mlflow.sklearn.log_model(model, "model")

    # ========================
    # WAJIB (LOCAL FOR DOCKER)
    # ========================
    mlflow.sklearn.save_model(model, "model")

    # ========================
    # SAVE RUN ID
    # ========================
    run_id = run.info.run_id
    print("RUN_ID:", run_id)

    with open("run_id.txt", "w") as f:
        f.write(run_id)

print("Training selesai")