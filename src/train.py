
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import build_preprocessor

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
COMPARISON_PATH = REPORT_DIR / "notebook_model_evaluation.csv"


def train_and_compare_models():
    # 1. Load cleaned dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run src/data_cleaning.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"customerID", "Churn"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 2. Separate features and target
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Churn column contains unexpected values.")

    # 3. Split once so every model uses the same test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Define all five classifiers
    models = {
        "Dummy Classifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}

    # 5. Train and evaluate every model
    for model_name, classifier in models.items():
        print(f"\nTraining: {model_name}")

        preprocessor, _ = build_preprocessor()

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        }

        results.append(metrics)
        fitted_models[model_name] = pipeline

        print(
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f} | "
            f"F1={metrics['F1_Score']:.4f} | "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}"
        )

    # 6. Save actual comparison metrics
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(COMPARISON_PATH, index=False)

    # 7. Select the highest-F1 model and save its complete pipeline
    best_model_name = results_df.loc[0, "Model"]
    best_pipeline = fitted_models[best_model_name]

    joblib.dump(best_pipeline, MODEL_PATH)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))

    print(f"\nBest model by test F1-score: {best_model_name}")
    print(f"Comparison saved to: {COMPARISON_PATH}")
    print(f"Selected model saved to: {MODEL_PATH}")
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    train_and_compare_models()
