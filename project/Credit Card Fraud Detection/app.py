from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "fraud_model.joblib"
CONFIG_PATH = ROOT / "model" / "config.json"

print(MODEL_PATH)

# Load the trained model once when the application starts.
model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

FEATURES = config["features"]
THRESHOLD = float(config["threshold"])


@app.get("/")
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        threshold=THRESHOLD
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object."}), 400

    try:
        values = []

        for feature in FEATURES:
            value = data.get(feature)

            if value is None or isinstance(value, bool):
                return jsonify({
                    "error": f"Missing or invalid value: {feature}"
                }), 400

            number = float(value)

            if not np.isfinite(number):
                return jsonify({
                    "error": f"{feature} must be a finite number."
                }), 400

            values.append(number)

        # Maintain exactly the same feature order used during training.
        transaction = pd.DataFrame([values], columns=FEATURES)

        probability = float(model.predict_proba(transaction)[0, 1])
        is_fraud = probability >= THRESHOLD

        return jsonify({
            "prediction": int(is_fraud),
            "label": "Potential Fraud" if is_fraud else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": round(THRESHOLD, 6),
            "message": (
                "Flag for further review."
                if is_fraud
                else "Not flagged by this model."
            )
        })

    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "error": "All feature values must be valid numbers."
        }), 400


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)