import pytest
from app.predict import app
import numpy as np

@pytest.fixture
def client():
    return app.test_client()

def test_ping(client):

    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json == {"status" : "ok"}

def test_about(client):
    response = client.get("/about")

    assert response.status_code == 200
    assert response.json == {
        "description": "Electric Car Purchase Prediction API"
    }

customer = {
    "age": 28,
    "annual_income_usd": 190400.0,
    "daily_commute_km": 20.6,
    "number_of_cars_owned": 2,
    "charging_stations_near_home": 10,
    "charging_stations_near_work": 4,
    "environmental_concern_level": 3.0,
    "gender": "Male",
    "city_type": "Urban",
    "current_car_type": "SUV",
    "home_charging_possible": "No",
    "subsidy_available": "No",
    "range_anxiety_level": "Low",
}

def test_predict(client):
    response = client.post("/predict", json = customer)

    assert response.status_code == 200

    data = response.get_json()

    assert "purchase_probability" in data
    assert "threshold" in data
    assert "prediction" in data
    assert "prediction_label" in data
    assert 0 <= data["purchase_probability"] <= 1

    assert data["threshold"] == pytest.approx(0.27)

    expected_prediction = int(data["purchase_probability"] >= data["threshold"])

    assert data["prediction"] == expected_prediction

    expected_label = ("Likely to Buy" if expected_prediction == 1 else "Unlikely to Buy")

    assert data["prediction_label"] == expected_label

def test_predict_missing_field(client):
    sample = customer.copy()
    sample.pop("age")

    response = client.post("/predict", json = sample)

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Missing required fields"
    assert "age" in data["missing_fields"]

def test_predict_non_string_category(client):
    sample = customer.copy()
    sample["gender"] = 123

    response = client.post("/predict", json=sample)

    assert response.status_code == 422

    data = response.get_json()

    assert data["error"] == "Invalid categorical values"
    assert "gender" in data["invalid_fields"]

def test_predict_unexpected_field(client):
    sample = customer.copy()
    sample["occupation"] = "Engineer"

    response = client.post("/predict", json=sample)

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Unexpected fields supplied"
    assert "occupation" in data["unexpected_fields"]

def test_predict_invalid_category(client):
    sample = customer.copy()
    sample["gender"] = "Unknown"

    response = client.post("/predict", json=sample)

    assert response.status_code == 422

    data = response.get_json()

    assert data["error"] == "Invalid categorical values"
    assert "gender" in data["invalid_fields"]

def test_predict_invalid_numeric(client):
    sample = customer.copy()
    sample["age"] = "thirty"

    response = client.post("/predict", json=sample)

    assert response.status_code == 422

    data = response.get_json()

    assert data["error"] == "Invalid numerical values"
    assert "age" in data["invalid_fields"]

def test_predict_null_numeric(client):
    sample = customer.copy()
    sample["age"] = None

    response = client.post("/predict", json = sample)

    assert response.status_code == 422

    data = response.get_json()
    assert data["error"] == "Invalid numerical values"
    assert "age" in data["invalid_fields"]

def test_predict_invalid_numeric_values(client):
    sample = customer.copy()
    sample["age"] = -24
    sample["annual_income_usd"] = -1000000

    response = client.post("/predict", json = sample)

    assert response.status_code == 422

    data = response.get_json()
    assert data["error"] == "Numerical values outside allowed business rules"
    assert set(data["invalid_fields"]) == {"age", "annual_income_usd"}
  

def test_predict_boolean_numeric(client):
    sample = customer.copy()
    sample["age"] = True

    response = client.post("/predict", json = sample)

    assert response.status_code == 422

    data = response.get_json()
    assert data["error"] == "Invalid numerical values"
    assert "age" in data["invalid_fields"]

def test_predict_nan_numeric(client):
    sample = customer.copy()
    sample["age"] = np.nan
    response = client.post("/predict", json = sample)

    assert response.status_code == 422

    data = response.get_json()
    assert data["error"] == "Invalid numerical values"
    assert "age" in data["invalid_fields"]

def test_predict_infinite_numeric(client):
    sample = customer.copy()
    sample["age"] = float("inf")
    response = client.post("/predict", json = sample)

    assert response.status_code == 422

    data = response.get_json()
    assert data["error"] == "Invalid numerical values"
    assert "age" in data["invalid_fields"]

def test_predict_empty_json(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400

    data = response.get_json()
    assert data["error"] == "Missing required fields"

def test_predict_multiple_missing_fields(client):
    sample = customer.copy()

    sample.pop("age")
    sample.pop("gender")
    sample.pop("city_type")

    response = client.post("/predict", json=sample)

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Missing required fields"
    assert set(data["missing_fields"]) == {"age", "gender", "city_type"}


def test_predict_wrong_content_type(client):
    response = client.post(
        "/predict",
        data="This is not JSON",
        content_type="text/plain"
    )

    assert response.status_code == 415

def test_predict_non_object_json(client):
    response = client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Request body must be a JSON object"


