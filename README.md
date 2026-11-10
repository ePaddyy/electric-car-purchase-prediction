# Electric Car Purchase Prediction

A machine learning project that predicts the likelihood of a customer purchasing an electric vehicle based on demographic, financial, commuting, and charging-related information.

The project covers the machine-learning workflow, model serialization, API development, input validation, automated testing, and containerization with Docker.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Dataset and Features](#dataset-and-features)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Serialization](#model-serialization)
- [Flask Prediction API](#flask-prediction-api)
- [Input Validation](#input-validation)
- [Testing](#testing)
- [Dockerization](#dockerization)
- [Running the Project Locally](#running-the-project-locally)
- [Making Predictions](#making-predictions)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## Project Overview

Electric vehicle adoption is influenced by several factors, including income, commuting patterns, charging infrastructure, environmental awareness, and government incentives.

This project uses machine learning to estimate whether a customer is likely to purchase an electric vehicle based on these characteristics.

Beyond model development, the project demonstrates how a trained machine learning model can be packaged and exposed through a REST API, allowing other applications to consume its predictions.

## Objectives

The main objectives are to:

1. Explore and prepare customer data for machine learning.
2. Train and evaluate classification models.
3. Select an appropriate model and prediction threshold.
4. Save the trained model and its associated configuration.
5. Build a Flask API to serve predictions.
6. Implement input validation and meaningful error responses.
7. Test the API using Python and automated tests.
8. Containerize the application using Docker.
9. Deploy the API to a cloud hosting platform.

## Project Structure

```text
electric-car-purchase-prediction/
│
├── app/
│   ├── __pycache__/
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

### Directory description

| Directory/File | Purpose |
|---|---|
| `data/` | Contains the training, testing, and sample submission datasets. |
| `notebooks/` | Contains exploratory analysis and model development. |
| `scripts/` | Contains the model training script. |
| `models/` | Stores the serialized machine learning model. |
| `app/` | Contains the Flask prediction API. |
| `customer/` | Contains sample customer data for prediction requests. |
| `test/` | Contains API testing scripts. |
| `Dockerfile` | Defines the Docker image and application startup process. |
| `Pipfile` | Defines the project's Python dependencies. |
| `Pipfile.lock` | Locks dependency versions for reproducible installation. |
| `requirements.txt` | Provides a pip-compatible dependency list. |

## Dataset and Features

The project uses customer information to predict electric vehicle purchase behaviour.

The customer attributes currently used in the prediction request include:

| Feature | Description |
|---|---|
| `age` | Customer's age. |
| `annual_income_usd` | Customer's annual income in USD. |
| `daily_commute_km` | Average daily commuting distance. |
| `number_of_cars_owned` | Number of cars owned by the customer. |
| `charging_stations_near_home` | Charging stations available near the customer's home. |
| `charging_stations_near_work` | Charging stations available near the customer's workplace. |
| `environmental_concern_level` | Customer's level of environmental concern. |
| `gender` | Customer's gender category in the dataset. |
| `city_type` | Type of city where the customer lives. |
| `current_car_type` | Customer's current vehicle category. |
| `home_charging_possible` | Whether home charging is possible. |
| `subsidy_available` | Whether a purchase subsidy is available. |
| `range_anxiety_level` | Customer's level of concern about vehicle range. |

**Prediction target:** Electric vehicle purchase outcome. The exact target-column name and class encoding should be documented from the training dataset.

## Machine Learning Workflow

The machine learning development process is separated from the API application.

The notebook is used for experimentation and analysis, while `scripts/train.py` provides a dedicated location for the training workflow.

The general workflow includes:

1. Loading and examining the dataset.
2. Preparing customer features.
3. Training classification models.
4. Evaluating model performance.
5. Selecting the preferred model and decision threshold.
6. Saving the resulting model artifact.

The model evaluation metrics, final algorithm, and selected threshold will be added here once the final training results are documented.

## Model Serialization

The trained model is stored in:

```text
models/electric_car_model.joblib
```

The project uses **Joblib** to serialize the trained model artifact.

Joblib allows the application to load the trained model without having to retrain it whenever the API starts.

The API loads the saved artifact during application initialization and uses it to generate predictions.

## Flask Prediction API

The application exposes the trained machine learning model through a Flask API.

The API is located at:

```text
app/predict.py
```

Gunicorn is used as the production application server inside the Docker container.

The API accepts customer information as JSON and processes the request before passing valid input to the trained model.

### Prediction endpoint

```http
POST /predict
```

The endpoint receives customer information and returns a prediction response.

The API is designed to separate three responsibilities:

- Receiving prediction requests.
- Validating incoming customer information.
- Generating predictions using the saved model.

## Input Validation

Input validation is an important part of the API because incoming requests cannot always be assumed to contain valid customer information.

The application includes validation logic to check incoming customer data against defined rules.

The validation process is designed to:

- Identify missing or invalid input.
- Check customer values against the applicable validation rules.
- Collect multiple validation errors rather than stopping at the first error.
- Return meaningful error information.
- Prevent invalid requests from reaching the prediction stage.

This makes the API more reliable and easier for other applications to consume.

## Testing

The project contains a dedicated testing directory:

```text
test/
├── test.py
└── test_api.py
```

### Python API client

`test.py` acts as a client that sends sample customer information to the running API.

It uses the Python `requests` library to send HTTP requests and inspect the API responses.

### Automated API tests

`test_api.py` is reserved for automated API testing.

The testing process should cover:

- Valid customer information.
- Missing required fields.
- Invalid field values.
- Multiple validation errors.
- Appropriate HTTP response codes.
- Successful prediction responses.

### Running the client test

From the project root:

```bash
python -m test.test
```

The API must already be running for the client request to succeed.

## Dockerization

The application has been containerized using Docker.

The Dockerfile uses Python 3.11 as its base image and installs the dependencies required by the application.

It also installs `libgomp1`, which is needed by certain machine-learning libraries.

The container setup:

1. Uses a lightweight Python image.
2. Installs the required system dependency.
3. Installs Pipenv.
4. Copies the Pipfile and lockfile.
5. Installs locked Python dependencies.
6. Copies the Flask application and model artifact.
7. Exposes port `8000`.
8. Starts the application using Gunicorn.

### Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "app.predict:app"]
```

### Build the Docker image

Run the following command from the project root:

```bash
docker build -t ev-prediction .
```

### Run the container

```bash
docker run -it --rm -p 8000:8000 ev-prediction
```

The API will be accessible locally at:

```text
http://localhost:8000
```

The Docker container runs the Flask application using Gunicorn. Therefore, you do not need to start another local Flask server while the container is running.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Enter the project directory

```bash
cd electric-car-purchase-prediction
```

### 3. Install dependencies

Using Pipenv:

```bash
pip install pipenv
pipenv install
```

### 4. Start the application

For local development, start the Flask application using the development command configured in `app/predict.py`.

For containerized execution, build and run the Docker image using the commands in the Dockerization section.

## Making Predictions

Once the Docker container is running, open another terminal and execute:

```bash
python -m test.test
```

The test script sends the sample customer information to the API.

The API processes the request and returns the prediction response.

This demonstrates how an external Python client can communicate with a machine learning application through HTTP.

The same API can later be consumed by a frontend application or another service.

## Deployment

The application currently runs locally inside a Docker container.

The next stage is deploying the containerized application to a cloud hosting platform so that it can be accessed through a public URL.

**Planned hosting platform:** Render.

The deployment process will involve:

1. Pushing the project to GitHub.
2. Connecting the repository to Render.
3. Configuring a Docker Web Service.
4. Building and deploying the Docker image.
5. Testing the public API endpoint.
6. Updating the Python client to communicate with the deployed API.

Cloud deployment is still pending.

## Technologies Used

- **Python 3.11** – Main programming language.
- **Pandas and NumPy** – Data manipulation and numerical operations.
- **Scikit-learn and other selected ML libraries** – Machine learning development.
- **Joblib** – Model serialization.
- **Flask** – API development.
- **Gunicorn** – Production application server.
- **Requests** – HTTP client for API testing.
- **Pytest** – Automated testing.
- **Docker** – Application containerization.
- **Pipenv** – Dependency and environment management.
- **Git and GitHub** – Version control and source code hosting.

## Future Improvements

- Complete and document automated API test coverage.
- Improve API error handling and response consistency.
- Deploy the Dockerized API to Render.
- Add API documentation.
- Add a simple frontend for submitting customer information.
- Monitor API performance and prediction behaviour.
- Document final model evaluation results and the selected decision threshold.

## Project Status

| Component | Status |
|---|---|
| Dataset and model development | Implemented |
| Training script | Available |
| Model serialization | Implemented |
| Flask API | Implemented |
| Customer input validation | Implemented; multiple-error handling being refined |
| Python API client | Working locally |
| Automated API tests | Testing in progress |
| Docker image | Built successfully |
| Local Docker execution | Working |
| Cloud deployment | Pending |

---

**Author:** Emmanuel Paddy Adams

**Project:** Electric Car Purchase Prediction