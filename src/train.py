
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


def train_best_model():
    # 1. Load the cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Separate input features and target
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # 3. Split data before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Create preprocessing steps
    preprocessor, _ = build_preprocessor()

    # 5. Build a complete pipeline
    final_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # 6. Train using training data only
    final_pipeline.fit(X_train, y_train)

    # 7. Evaluate on the held-out test data
    predictions = final_pipeline.predict(X_test)
    probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    print("\nFINAL MODEL: LOGISTIC REGRESSION")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "F1 Score:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    # 8. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Saved location:", MODEL_PATH)


if __name__ == "__main__":
    train_best_model()
