#Goal: Dataset-la irukkura column names, data types, missing values, duplicate rows, target column ellathaiyum check pannradhu

from pathlib import Path
import pandas as pd

# 1. Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Locate the original dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def validate_dataset():
    """Load the dataset and perform basic data-quality checks."""

    # 3. Check whether the dataset file exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # 4. Load the CSV file
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("CUSTOMER INTELLIGENCE PLATFORM - DATA VALIDATION")
    print("=" * 60)

    # 5. Display dataset dimensions
    print("\n1. Dataset shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 6. Display column names
    print("\n2. Column names:")
    print(df.columns.tolist())

    # 7. Display data types and non-null counts
    print("\n3. Dataset information:")
    df.info()

    # 8. Count missing values
    print("\n4. Missing values by column:")
    print(df.isnull().sum())

    # 9. Count duplicate rows
    print("\n5. Duplicate rows:")
    print(df.duplicated().sum())

    # 10. Check the target column
    print("\n6. Target column validation:")
    if "Churn" in df.columns:
        print("Churn column exists.")
        print(df["Churn"].value_counts(dropna=False))
    else:
        print("ERROR: Churn column is missing.")

    # 11. Check unique customer IDs
    print("\n7. Customer ID validation:")
    if "customerID" in df.columns:
        print("Unique customer IDs:", df["customerID"].nunique())
        print("Missing customer IDs:", df["customerID"].isnull().sum())
    else:
        print("ERROR: customerID column is missing.")

    print("\nData validation completed.")


if __name__ == "__main__":
    validate_dataset()
