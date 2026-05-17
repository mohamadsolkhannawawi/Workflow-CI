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

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='insurance_preprocessing/train.csv')
parser.add_argument('--test_data', type=str, default='insurance_preprocessing/test.csv')
args = parser.parse_args()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("insurance_prediction")

os.makedirs("artifacts", exist_ok=True)

train = pd.read_csv(args.train_data)
test = pd.read_csv(args.test_data)

X_train = train.drop('charges', axis=1)
y_train = train['charges']
X_test = test.drop('charges', axis=1)
y_test = test['charges']

with mlflow.start_run() as run:

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "model")

    # 🔥 simpan run_id ke file
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Training selesai")