
import os

import joblib
import pandas as pd

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 23
THRESHOLD = 0.27



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"

# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.lower()

# Separate target
y = (df["will_buy_ev"] == "Yes").astype(int)

# Remove target and ID from features
X = df.drop(columns=["will_buy_ev", "id"])

# Preserve feature names for inference
feature_columns = X.columns.tolist()

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Convert DataFrames into dictionaries
X_train_full_dict = X_train_full.to_dict(orient="records")
X_test_dict = X_test.to_dict(orient="records")

# --------------------------------------------------
# 4. Build the LightGBM pipeline
# --------------------------------------------------

def make_pipe_lgbm(
    n_estimators=200,
    max_depth=3,
    num_leaves=15,
    learning_rate=0.05
):
    return make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    )


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

lgbm = make_pipe_lgbm()

lgbm.fit(X_train_full_dict, y_train_full)

# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

test_probabilities = lgbm.predict_proba(X_test_dict)[:, 1]

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)

test_auc = roc_auc_score(y_test, test_probabilities)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

conf_matrix = confusion_matrix(y_test, test_predictions)

print("\nTEST RESULTS")
print("-" * 40)
print(f"ROC-AUC:   {test_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# --------------------------------------------------
# 7. Package everything
# --------------------------------------------------

model_artifact = {
    "model": lgbm,
    "model_name": "ECPP-LightGBM",
    "threshold": THRESHOLD,
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 3,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "random_state": 42
    },
    "feature_columns": feature_columns,
    "target": "will_buy_ev",
    "metrics": {
        "test_auc": test_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "confusion_matrix": conf_matrix.tolist()
    }
}

# --------------------------------------------------
# 8. Save everything into one Joblib file
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "electric_car_model.joblib"

joblib.dump(
    model_artifact,
    MODEL_PATH,
    compress=3
)

print(f"\nModel artifact saved successfully: {MODEL_PATH}")

# --------------------------------------------------
# 9. Verify saved artifact
# --------------------------------------------------

loaded_artifact = joblib.load(MODEL_PATH)

print("\nVERIFICATION")
print("-" * 40)
print("Model:", loaded_artifact["model_name"])
print("Threshold:", loaded_artifact["threshold"])
print("Hyperparameters:", loaded_artifact["hyperparameters"])
print("Test AUC:", loaded_artifact["metrics"]["test_auc"])

print("\nModel successfully saved and loaded!")

