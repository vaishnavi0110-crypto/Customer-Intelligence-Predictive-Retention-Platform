
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def inspect_features():
    df = pd.read_csv(DATA_PATH)

    # Target we want the model to predict
    target = "Churn"

    # Columns that should not be used as predictive inputs
    excluded_columns = ["customerID", target]

    # Input features
    X = df.drop(columns=excluded_columns)

    # Prediction target
    y = df[target].map({"No": 0, "Yes": 1})

    print("=" * 50)
    print("FEATURE ENGINEERING - INITIAL INSPECTION")
    print("=" * 50)

    print("\nInput feature count:", X.shape[1])
    print("Target column:", target)

    print("\nInput feature columns:")
    print(X.columns.tolist())

    print("\nTarget values:")
    print(y.value_counts().sort_index())

    print("\nTarget mapping:")
    print("No = 0 (customer retained)")
    print("Yes = 1 (customer churned)")

    print("\nMissing target values after mapping:", y.isnull().sum())

    print("\nFeature data types:")
    print(X.dtypes)


if __name__ == "__main__":
    inspect_features()
