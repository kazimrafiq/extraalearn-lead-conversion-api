
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "final_lead_conversion_model.joblib"
)
model = joblib.load(MODEL_PATH)

REQUIRED_FEATURES = [
    "age", "current_occupation", "first_interaction",
    "profile_completed", "website_visits",
    "time_spent_on_website", "page_views_per_visit",
    "last_activity", "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral"
]

NUMERICAL_FEATURES = [
    "age", "website_visits",
    "time_spent_on_website", "page_views_per_visit"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExtraaLearn Lead Conversion API",
        "endpoints": ["/health", "/predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing
        }), 400

    try:
        input_df = pd.DataFrame([
            {f: data[f] for f in REQUIRED_FEATURES}
        ])

        if input_df.isnull().any().any():
            return jsonify({
                "error": "Feature values cannot be null."
            }), 400

        for feature in NUMERICAL_FEATURES:
            input_df[feature] = pd.to_numeric(
                input_df[feature], errors="raise"
            )

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            "prediction": prediction,
            "prediction_label": (
                "Converted" if prediction == 1
                else "Not Converted"
            ),
            "conversion_probability": round(probability, 4)
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Invalid feature value.",
            "details": str(error)
        }), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
