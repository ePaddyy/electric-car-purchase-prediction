FROM python:3.11-slim

# Install OpenMP runtime for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Pipenv
RUN pip install --no-cache-dir pipenv

WORKDIR /app

# Copy dependency files
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy

# Copy application and model files
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 10000

ENTRYPOINT ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} app.predict:app"]