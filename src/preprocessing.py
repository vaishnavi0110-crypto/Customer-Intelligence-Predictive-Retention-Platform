
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def build_preprocessor():
    # 2. Load the cleaned data
    df = pd.read_csv(DATA_PATH)

    # 3. Separate input features and target
    X = df.drop(columns=["customerID", "Churn"])

    # 4. Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # 5. Numeric preprocessing steps
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 6. Categorical preprocessing steps
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 7. Apply the correct pipeline to each column type
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    print("Numeric columns:", numeric_columns)
    print("\nCategorical columns:", categorical_columns)

    return preprocessor, X


if __name__ == "__main__":
    preprocessor, X = build_preprocessor()

    # 8. Fit and transform for this initial pipeline check
    X_transformed = preprocessor.fit_transform(X)

    print("\nOriginal feature count:", X.shape[1])
    print("Transformed feature count:", X_transformed.shape[1])
    print("Preprocessing pipeline works successfully!")
