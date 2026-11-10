
import requests

url = "https://electric-car-purchase-prediction.onrender.com/predict"

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
    "range_anxiety_level": "Low"
}

response = requests.post(url, json=customer, timeout=60)

print("Status:", response.status_code)
print("Prediction:", response.json())
