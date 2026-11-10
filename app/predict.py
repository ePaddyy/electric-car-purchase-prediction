
import numpy as np
import joblib

from pathlib import Path
from flask import Flask, jsonify, request


# Load model

base_dir = Path(__file__).resolve().parent.parent

model_path = base_dir / "models" / "electric_car_model.joblib"
artifact = joblib.load(model_path)

model = artifact["model"]
THRESHOLD = artifact["threshold"]


# Define feature groups

CATEGORICAL_VALUES = {
    "gender": {"Male", "Female", "Other"},
    "city_type": {"Suburban", "Rural", "Urban"},
    "current_car_type": {"Sedan", "SUV", "Hatchback", "Truck"},
    "home_charging_possible": {"Yes", "No"},
    "subsidy_available": {"No", "Yes"},
    "range_anxiety_level": {"Low", "Medium", "High"}
}

NUMERIC_FIELDS = {
    "age",
    "annual_income_usd",
    "daily_commute_km",
    "number_of_cars_owned",
    "charging_stations_near_home",
    "charging_stations_near_work",
    "environmental_concern_level"
}

NUMERIC_RULES = {
    "age": {
        "min": 0,
        "max": 100,
        "integer": True
    },
    "annual_income_usd": {
        "min": 0
    },
    "daily_commute_km": {
        "min": 0
    },
    "number_of_cars_owned": {
        "min": 0,
        "integer": True
    },
    "charging_stations_near_home": {
        "min": 0,
        "integer": True
    },
    "charging_stations_near_work": {
        "min": 0,
        "integer": True
    },
    "environmental_concern_level": {
        "min": 1,
        "max": 50,
        "integer": True
    }
}

CATEGORICAL_FIELDS = set(CATEGORICAL_VALUES.keys())

REQUIRED_FIELDS = NUMERIC_FIELDS | CATEGORICAL_FIELDS


# Create Flask application

app = Flask("electric-car-purchase-prediction")




@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})



@app.route("/about", methods=["GET"])
def about():
    return jsonify({
        "description": "Electric Car Purchase Prediction API"
    })


# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 415

    customer = request.get_json(silent=True)

    if not isinstance(customer, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400


    missing = REQUIRED_FIELDS - set(customer.keys())

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": list(missing)
        }), 400

    unexpected = set(customer.keys()) - REQUIRED_FIELDS

    if unexpected:
        return jsonify({
            "error": "Unexpected fields supplied",
            "unexpected_fields": list(unexpected)
        }), 400

    # Validate categorical features


    invalid_categories = {}

    for field in CATEGORICAL_FIELDS:

        value = customer[field]

        if not isinstance(value, str):
            invalid_categories[field] = value

        elif value not in CATEGORICAL_VALUES[field]:
            invalid_categories[field] = value

    if invalid_categories:
        return jsonify({
            "error": "Invalid categorical values",
            "invalid_fields": invalid_categories,
            "allowed_values": {
                field: list(CATEGORICAL_VALUES[field])
                for field in invalid_categories
            }
        }), 422



    # Validate numerical features
    
    invalid_numerics = {}

    for field in NUMERIC_FIELDS:

        value = customer[field]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_numerics[field] = value

        elif not np.isfinite(value):
            invalid_numerics[field] = value

    if invalid_numerics:
        return jsonify({
            "error": "Invalid numerical values",
            "invalid_fields": invalid_numerics
        }), 422

    # Validate numerical business rules
    invalid_ranges = {}

    for field, rules in NUMERIC_RULES.items():
        value = customer[field]

        # Check minimum value
        if rules.get("min") is not None and value < rules["min"]:
            invalid_ranges[field] = value

        # Check maximum value
        if rules.get("max") is not None and value > rules["max"]:
            invalid_ranges[field] = value

        # Check integer requirement
        if rules.get("integer") and value % 1 != 0:
            invalid_ranges[field] = value

    if invalid_ranges:
        return jsonify({
            "error": "Numerical values outside allowed business rules",
            "invalid_fields": invalid_ranges
        }), 422
    
    # Generate prediction

    probability = float(model.predict_proba([customer])[0, 1])

    prediction = probability >= THRESHOLD

    return jsonify({
        "purchase_probability": probability,
        "threshold": THRESHOLD,
        "prediction": int(prediction),
        "prediction_label": (
            "Likely to Buy" if prediction else "Unlikely to Buy"
        )
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
