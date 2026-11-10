# Electric Car Purchase Prediction

A machine learning project that predicts whether a customer is likely to purchase an electric vehicle (EV) based on demographic, financial, lifestyle, and charging-infrastructure features.

The project covers the end-to-end machine learning workflow, from exploratory data analysis and model development to API development, automated testing, Docker containerization, and cloud deployment.

**GitHub Repository:** [electric-car-purchase-prediction](https://github.com/ePaddyy/electric-car-purchase-prediction)

**Live API:** [Electric Car Purchase Prediction API](https://electric-car-purchase-prediction.onrender.com)

**Cloud Platform:** Render

**Model:** LightGBM Classifier

**API Framework:** Flask

---

## 1. Project Objectives

The main objectives are to:

- Explore customer characteristics associated with electric vehicle purchases.
- Perform data cleaning, preprocessing, and exploratory data analysis.
- Engineer useful features for model development.
- Train and evaluate multiple classification algorithms.
- Select and serialize a machine learning model.
- Build a REST API to serve predictions.
- Validate incoming customer data.
- Write automated tests for API functionality.
- Containerize the application using Docker.
- Deploy the API to the cloud using Render.

## 2. Project Structure

```text
electric-car-purchase-prediction/
│
├── app/
│   └── predict.py
│
├── customer/
│   ├── __init__.py
│   └── customer.py
│
├── data/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── models/
│   └── electric_car_model.joblib
│
├── notebooks/
│   ├── catboost_info/
│   └── electric_car_purchase_predictions.ipynb
│
├── scripts/
│   └── train.py
│
├── test/
│   ├── __init__.py
│   ├── test.py
│   ├── test_api.py
│   └── test_online.py
│
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── requirements.txt
└── README.md
```

### Directory descriptions

| Directory/File | Purpose |
|---|---|
| `app/` | Flask prediction API |
| `customer/` | Sample customer input data |
| `data/` | Training and testing datasets |
| `models/` | Serialized machine learning model |
| `notebooks/` | EDA, experimentation, and model evaluation |
| `scripts/` | Model training and serialization |
| `test/` | API and prediction testing |
| `Dockerfile` | Docker image configuration |
| `Pipfile` | Python dependency configuration |
| `Pipfile.lock` | Locked dependency versions |
| `requirements.txt` | Python package requirements |

## 3. Dataset and Features

The project predicts the target variable:

- `will_buy_ev`: Whether a customer will purchase an electric vehicle.

The model uses the following features:

| Feature | Description |
|---|---|
| age | Customer's age |
| annual_income_usd | Annual income in USD |
| daily_commute_km | Daily commuting distance |
| number_of_cars_owned | Number of cars owned |
| charging_stations_near_home | Charging stations near home |
| charging_stations_near_work | Charging stations near work |
| environmental_concern_level | Environmental concern score |
| gender | Customer's gender |
| city_type | Type of city |
| current_car_type | Current vehicle category |
| home_charging_possible | Availability of home charging |
| subsidy_available | Availability of EV subsidies |
| range_anxiety_level | Customer's range anxiety level |

## 4. Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset, examine feature distributions, identify relationships, and investigate the target variable.

Key activities included:

- Numerical and categorical feature analysis.
- Target distribution analysis.
- Missing-value and data-quality checks.
- Feature relationship analysis.
- Mutual information analysis.
- Visualization of important patterns.

The analysis indicated that environmental concern, annual income, and subsidy availability were useful features for understanding EV purchase patterns.

These relationships are observational and do not establish causation.

## 5. Model Development

Multiple classification algorithms were explored and evaluated.

Models included:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

Model comparison was based primarily on ROC-AUC, alongside other classification metrics.

### Model comparison

| Model | Validation ROC-AUC |
|---|---:|
| XGBoost | 0.934415 |
| LightGBM | 0.934368 |
| CatBoost | 0.933152 |
| Logistic Regression | 0.932669 |
| Random Forest | 0.929537 |
| Extra Trees | 0.928865 |
| Decision Tree | 0.912376 |

LightGBM was selected for the final model.

## 6. Final Model Configuration

The selected LightGBM configuration:

| Parameter | Value |
|---|---:|
| Model | ECPP-LightGBM |
| n_estimators | 200 |
| max_depth | 3 |
| num_leaves | 15 |
| learning_rate | 0.05 |
| random_state | 42 |
| Classification threshold | 0.27 |

The model artifact is saved using Joblib:

```text
models/electric_car_model.joblib
```

The artifact stores the trained model and prediction configuration used by the API.

### Training script

The `scripts/train.py` script trains the selected model configuration and saves the resulting artifact.

Run it from the project root:

```bash
python scripts/train.py
```

The notebook contains the experimentation, model comparison, and final evaluation workflow. The training script is used to train and serialize the selected model configuration. Their reported metrics should be interpreted according to their respective evaluation datasets and procedures.

### Training script's reported evaluation output

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9395 |
| Precision | 0.6003 |
| Recall | 0.8529 |
| F1 Score | 0.7046 |

These are the metrics reported by the training script, not a claim that they represent the same evaluation as the notebook's final test-set results.

## 7. API Development

A Flask REST API was developed to expose the trained model for external predictions.

The API loads the serialized model artifact and provides endpoints for checking service availability, retrieving model information, and generating predictions.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ping` | GET | Checks whether the API is running |
| `/about` | GET | Returns model information |
| `/predict` | POST | Generates an EV purchase prediction |

### Input validation

The prediction endpoint validates incoming requests before passing data to the model.

Validation includes:

- Checking that the request uses JSON.
- Ensuring the request body is a JSON object.
- Checking for missing required fields.
- Rejecting unexpected fields.
- Validating categorical feature values.
- Validating numerical data types.
- Checking numerical business rules and permitted ranges.

The API returns appropriate HTTP error responses for invalid requests.

## 8. Testing

The project includes three testing scripts:

| File | Purpose |
|---|---|
| `test/test.py` | Sends sample customer data to the API |
| `test/test_api.py` | Automated API tests using Pytest |
| `test/test_online.py` | Tests predictions against the deployed Render API |

Run the automated API tests:

```bash
pytest test/test_api.py
```

Run the local prediction client:

```bash
python -m test.test
```

Run the online prediction client:

```bash
python -m test.test_online
```

The online test successfully sends customer information to the deployed API and receives a prediction response.

## 9. Docker Containerization

Docker was used to package the Flask application and its model dependencies into a portable container.

The Dockerfile:

- Uses Python 3.11 slim as the base image.
- Installs the required system dependency `libgomp1`.
- Installs Python dependencies using Pipenv.
- Copies the application and serialized model.
- Starts the application using Gunicorn.

### Build the Docker image

```bash
docker build -t ev-prediction .
```

### Run the container locally

```bash
docker run --rm -p 8000:10000 ev-prediction
```

The API can then be accessed locally at:

```text
http://localhost:8000
```

## 10. Cloud Deployment — Render

The application has been successfully deployed to **Render** as a Docker-based Web Service.

Render builds the application from the GitHub repository using the project's Dockerfile and runs the container in the cloud.

### Deployment configuration

| Setting | Configuration |
|---|---|
| Platform | Render |
| Service Type | Web Service |
| Runtime | Docker |
| Repository | ePaddyy/electric-car-purchase-prediction |
| Branch | main |
| Dockerfile | `./Dockerfile` |
| Health Check Path | `/ping` |
| Application Server | Gunicorn |

### Port configuration

Render provides the application port through the `PORT` environment variable.

The Dockerfile uses:

```dockerfile
EXPOSE 10000

ENTRYPOINT ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} app.predict:app"]
```

This allows Gunicorn to bind to the port supplied by Render, with `10000` as the default.

### Live API

**Base URL:** https://electric-car-purchase-prediction.onrender.com

Available endpoints:

```text
GET  https://electric-car-purchase-prediction.onrender.com/ping
GET  https://electric-car-purchase-prediction.onrender.com/about
POST https://electric-car-purchase-prediction.onrender.com/predict
```

The deployed `/ping` endpoint has been verified, and the `/predict` endpoint has successfully generated predictions through the online testing client.

### Making predictions against the deployed API

External applications can send customer information to the deployed `/predict` endpoint using an HTTP POST request.

Example using Python:

```python
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

response = requests.post(
    url,
    json=customer,
    timeout=60
)

print(response.status_code)
print(response.json())
```

The request uses JSON, and the API validates the supplied features before generating a prediction.

**Note:** The first request after a period of inactivity may take longer on a free hosting instance because the service may need to wake up.

## 11. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, LightGBM, XGBoost, CatBoost |
| Model Serialization | Joblib |
| API Development | Flask |
| API Server | Gunicorn |
| Testing | Pytest, Requests |
| Containerization | Docker |
| Cloud Deployment | Render |
| Version Control | Git, GitHub |

## 12. Future Improvements

Potential improvements include:

- Build a simple frontend for interactive predictions.
- Add API authentication and request protection.
- Improve API monitoring and logging.
- Add model performance monitoring.
- Implement CI/CD checks.
- Explore additional model explainability techniques.
- Improve the prediction interface and documentation.

## 13. Project Status

| Component | Status |
|---|---|
| Data cleaning and EDA | Completed |
| Feature engineering | Completed |
| Model experimentation | Completed |
| Final model training | Completed |
| Model serialization | Completed |
| Flask API development | Completed |
| Input validation | Implemented |
| Automated API testing | Tested |
| Docker containerization | Completed |
| Render cloud deployment | Deployed |
| `/ping` health check | Verified |
| `/predict` cloud endpoint | Verified |

---

**Author:** Emmanuel Paddy Adams

**GitHub:** [ePaddyy](https://github.com/ePaddyy)