# Electric Car Purchase Prediction

A machine learning project that predicts whether a customer is likely to purchase an electric vehicle based on demographic, financial, commuting, and charging-infrastructure characteristics.

This project covers the end-to-end machine learning workflow, from exploratory data analysis and model experimentation to model serialization, Flask API development, input validation, testing, and Docker containerization.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Dataset and Features](#dataset-and-features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Final Model Training and Serialization](#final-model-training-and-serialization)
- [Flask Prediction API](#flask-prediction-api)
- [Input Validation](#input-validation)
- [Testing](#testing)
- [Dockerization](#dockerization)
- [Running the Project](#running-the-project)
- [Making Predictions](#making-predictions)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## Project Overview

Electric vehicle adoption is influenced by several factors, including income, commuting patterns, charging infrastructure, environmental awareness, and government incentives.

This project applies machine learning classification techniques to estimate the likelihood of a customer purchasing an electric vehicle based on these characteristics.

Beyond model development, the project demonstrates how a trained machine learning model can be packaged and exposed through a REST API, allowing external applications to request predictions.

## Objectives

The main objectives are to:

1. Explore and understand customer data through exploratory data analysis.
2. Prepare numerical and categorical features for machine learning.
3. Experiment with different classification algorithms.
4. Tune model hyperparameters and compare model performance.
5. Select an appropriate model and classification threshold.
6. Train the selected model on the full training dataset.
7. Serialize the trained model and its configuration.
8. Develop a Flask API for serving predictions.
9. Implement input validation and meaningful error responses.
10. Test the API and containerize the application using Docker.
11. Prepare the application for cloud deployment.

## Project Structure

```text
electric-car-purchase-prediction/
│
├── app/
│   └── predict.py
│
├── customer/
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
│   └── electric_car_purchase_predictions.ipynb
│
├── scripts/
│   └── train.py
│
├── test/
│   ├── test.py
│   └── test_api.py
│
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── requirements.txt
└── README.md
```

### Directory Description

| Directory/File | Purpose |
|---|---|
| `data/` | Contains the training, testing, and sample submission datasets. |
| `notebooks/` | Contains exploratory analysis, model experimentation, evaluation, and model selection. |
| `scripts/` | Contains the final model training script. |
| `models/` | Stores the serialized machine learning model artifact. |
| `app/` | Contains the Flask prediction API. |
| `customer/` | Contains sample customer information for prediction requests. |
| `test/` | Contains scripts for testing the API. |
| `Dockerfile` | Defines the Docker image and application startup configuration. |
| `Pipfile` | Defines project dependencies. |
| `Pipfile.lock` | Locks dependency versions for reproducible installation. |
| `requirements.txt` | Provides a pip-compatible dependency list. |

## Dataset and Features

The project uses customer information to predict electric vehicle purchase behaviour.

The target variable is:

- `will_buy_ev`: Indicates whether a customer purchased an electric vehicle.
  - `1`: Yes
  - `0`: No

### Input Features

| Feature | Description |
|---|---|
| `age` | Customer's age. |
| `annual_income_usd` | Customer's annual income in USD. |
| `daily_commute_km` | Average daily commuting distance. |
| `number_of_cars_owned` | Number of cars owned by the customer. |
| `charging_stations_near_home` | Number of charging stations near the customer's home. |
| `charging_stations_near_work` | Number of charging stations near the customer's workplace. |
| `environmental_concern_level` | Customer's environmental concern level. |
| `gender` | Customer's gender category. |
| `city_type` | Type of city where the customer lives. |
| `current_car_type` | Customer's current vehicle category. |
| `home_charging_possible` | Whether home charging is possible. |
| `subsidy_available` | Whether a purchase subsidy is available. |
| `range_anxiety_level` | Customer's level of concern about vehicle range. |

## Exploratory Data Analysis

Exploratory data analysis was conducted in the project notebook to understand the dataset, examine feature distributions, investigate relationships between variables, and identify patterns associated with EV purchases.

The analysis included:

- Examining numerical and categorical features.
- Investigating the target variable distribution.
- Exploring relationships between customer characteristics and EV purchase behaviour.
- Evaluating feature associations using mutual information.
- Identifying potentially influential customer characteristics.

Some notable observations from the analysis include:

- Environmental concern had the highest mutual information score among numerical features.
- Annual income was another important numerical feature.
- Subsidy availability had the highest mutual information score among categorical features.
- Customers with home charging availability showed a higher observed EV purchase rate.
- Customers with available subsidies also showed a higher observed purchase rate.

These findings describe associations in the dataset and do not establish causal relationships.

## Model Development

The project follows an experimentation-based approach to model selection.

Multiple classification algorithms were investigated and compared using ROC-AUC as the primary model-selection metric.

### Models Evaluated

| Model |
|---|
| Logistic Regression |
| Decision Tree |
| Random Forest |
| Extra Trees |
| XGBoost |
| LightGBM |
| CatBoost |

Hyperparameter tuning was performed to identify suitable configurations for the candidate models.

### Validation Results

The notebook's model comparison produced the following validation ROC-AUC scores:

| Model | Validation ROC-AUC |
|---|---:|
| XGBoost | 0.934415 |
| LightGBM | 0.934368 |
| CatBoost | 0.933152 |
| Logistic Regression | 0.932669 |
| Random Forest | 0.929537 |
| Extra Trees | 0.928865 |
| Decision Tree | 0.912376 |

LightGBM was selected as the final model following the experimentation and threshold-selection process.

Although XGBoost achieved a marginally higher validation ROC-AUC, LightGBM was selected for the final implementation.

## Model Evaluation

The notebook uses the supplied test dataset as the final evaluation dataset after model experimentation and selection.

The evaluation process considers:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Confusion Matrix

The classification threshold is also considered when converting predicted probabilities into purchase classifications.

**Note:** Notebook evaluation results and final training-script results represent different stages of the workflow and should not be interpreted as measurements from the same fitted model on the same dataset.

## Final Model Training and Serialization

After selecting the model configuration, a separate training script was created:

```text
scripts/train.py
```

This script trains the selected LightGBM configuration using the full training dataset rather than repeating the notebook's experimental workflow.

### Selected Model

| Parameter | Value |
|---|---|
| Model | LightGBM Classifier |
| Model Name | ECPP-LightGBM |
| Number of Estimators | 200 |
| Maximum Depth | 3 |
| Number of Leaves | 15 |
| Learning Rate | 0.05 |
| Random State | 42 |
| Classification Threshold | 0.27 |

### Final Training Script Output

The training script reported the following results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9395 |
| Precision | 0.6003 |
| Recall | 0.8529 |
| F1 Score | 0.7046 |

Confusion Matrix:

```text
[[97085 13274]
 [ 3439 19935]]
```

These figures are recorded as the training script's reported evaluation output, separately from the notebook's final test evaluation.

### Model Serialization

The trained model and its associated configuration are saved using Joblib.

Artifact location:

```text
models/electric_car_model.joblib
```

The saved artifact includes the model and prediction configuration required by the API, including the selected classification threshold.

The training script also verifies that the saved artifact can be loaded successfully.

## Flask Prediction API

The trained model is exposed through a Flask REST API.

The API implementation is located at:

```text
app/predict.py
```

Gunicorn is used as the production application server.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/ping` | Checks API availability. |
| GET | `/about` | Provides information about the model. |
| POST | `/predict` | Receives customer information and returns a prediction. |

### Prediction Workflow

The prediction endpoint follows this general process:

1. Receives customer information in JSON format.
2. Checks the incoming request.
3. Validates the supplied customer attributes.
4. Processes valid input using the saved model.
5. Generates a prediction.
6. Returns the prediction response as JSON.

The API loads the serialized model artifact rather than retraining the model whenever a prediction request is received.

## Input Validation

The API includes input validation to ensure that incoming customer information meets the expected requirements.

Validation covers:

- Required customer fields.
- Numerical and categorical data types.
- Allowed categorical values.
- Numerical value constraints.
- Business rules for customer information.

Invalid requests are rejected before prediction, with error responses intended to help identify the problematic fields.

The validation logic is also being refined to collect multiple validation errors in a single response rather than stopping after the first invalid field.

## Testing

The project includes a dedicated testing directory:

```text
test/
├── test.py
└── test_api.py
```

### Python API Client

`test.py` acts as an external client that sends sample customer information to the running API.

It uses the Python `requests` library to send HTTP requests and inspect the API responses.

Run it from the project root:

```bash
python -m test.test
```

### Automated API Testing

`test_api.py` is used for API testing.

The intended test coverage includes:

- Successful prediction requests.
- Missing required fields.
- Invalid numerical values.
- Invalid categorical values.
- Multiple validation errors.
- Appropriate HTTP status codes.
- Error handling for malformed requests.

## Dockerization

The application has been containerized using Docker to provide a consistent runtime environment.

The Dockerfile uses Python 3.11 as its base image.

It also installs `libgomp1`, a system dependency required by certain machine learning libraries.

### Docker Configuration

The Dockerfile performs the following operations:

1. Uses a lightweight Python 3.11 image.
2. Installs required system dependencies.
3. Installs Pipenv.
4. Copies the Pipfile and lockfile.
5. Installs locked Python dependencies.
6. Copies the Flask application and model artifact.
7. Exposes port `8000`.
8. Starts the application using Gunicorn.

### Build the Docker Image

From the project root:

```bash
docker build -t ev-prediction .
```

### Run the Container

```bash
docker run -it --rm -p 8000:8000 ev-prediction
```

The API will be accessible locally at:

```text
http://localhost:8000
```

The container starts the Flask application through Gunicorn. There is no need to start a second local Flask server while the Docker container is already running.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ePaddyy/electric-car-purchase-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd electric-car-purchase-prediction
```

### 3. Install Dependencies

Install Pipenv:

```bash
pip install pipenv
```

Install the project dependencies:

```bash
pipenv install
```

### 4. Build the Docker Image

```bash
docker build -t ev-prediction .
```

### 5. Start the Application

```bash
docker run -it --rm -p 8000:8000 ev-prediction
```

### 6. Send a Prediction Request

Open another terminal and run:

```bash
python -m test.test
```

The test client will send the sample customer information to the running API.

## Making Predictions

The API accepts customer information through a JSON request.

Example request structure:

```json
{
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
```

The request is sent to:

```http
POST /predict
```

The API validates the customer information and returns a prediction response.

The same endpoint can be consumed by other applications, including web frontends and external services.

## Deployment

The application currently runs locally inside a Docker container.

The next stage is to deploy the containerized application to a cloud hosting platform.

**Planned hosting platform:** Render.

The planned deployment workflow is:

1. Push the latest project changes to GitHub.
2. Connect the GitHub repository to Render.
3. Configure a Docker Web Service.
4. Build and deploy the Docker image.
5. Test the public API endpoints.
6. Update the Python client to communicate with the deployed API.

Cloud deployment is still pending.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and machine learning utilities |
| LightGBM | Final classification model |
| XGBoost | Model experimentation |
| CatBoost | Model experimentation |
| Joblib | Model serialization |
| Flask | REST API development |
| Gunicorn | Production application server |
| Requests | HTTP client and API testing |
| Pytest | Automated testing |
| Docker | Application containerization |
| Pipenv | Dependency management |
| Git and GitHub | Version control and source code hosting |

## Future Improvements

- Complete cloud deployment.
- Finalize multi-error validation handling.
- Expand automated API test coverage.
- Add API documentation.
- Develop a frontend for submitting customer information.
- Monitor API performance and prediction behaviour.
- Document and compare final model evaluation results.
- Add deployment and usage instructions for external API consumers.

## Project Status

| Component | Status |
|---|---|
| Exploratory data analysis | Completed |
| Model experimentation | Completed |
| Hyperparameter tuning | Completed |
| Model selection | Completed |
| Final model training script | Implemented |
| Model serialization | Implemented |
| Flask prediction API | Implemented |
| Input validation | Implemented; refinements ongoing |
| Python API client | Working locally |
| Automated API testing | In progress |
| Docker image | Built successfully |
| Local Docker execution | Working |
| Cloud deployment | Pending |

---

**Author:** Emmanuel Paddy Adams

**Repository:** [electric-car-purchase-prediction](https://github.com/ePaddyy/electric-car-purchase-prediction)