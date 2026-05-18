"""
Model Serving Script - Insurance Model
Dijalankan di dalam Docker container
"""

import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model dari folder model/
MODEL_PATH = "/app/model"
model = mlflow.sklearn.load_model(MODEL_PATH)
print(f"[INFO] Model loaded dari {MODEL_PATH}")


@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Support format MLflow: {"dataframe_records": [...]}
        if "dataframe_records" in data:
            df = pd.DataFrame(data["dataframe_records"])
        elif "instances" in data:
            df = pd.DataFrame(data["instances"])
        else:
            df = pd.DataFrame([data])

        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
