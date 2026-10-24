
from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

# Output directory
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Output validation report
OUTPUT_PATH = PROCESSED_DIR / "data_validation_report.csv"


def validate_dataset():
    """Validate dataset structure and basic data quality."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please check the CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("CUSTOMER INTELLIGENCE PLATFORM - DATA VALIDATION")
    print("=" * 60)

    print("\n1. Dataset shape")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. Column names")
    print(df.columns.tolist())

    print("\n3. Data types and non-null counts")
    df.info()

    print("\n4. Missing values by column")
    print(df.isnull().sum())

    print("\n5. Duplicate rows")
    duplicate_count = int(df.duplicated().sum())
    print(duplicate_count)

    print("\n6. Target column validation")
    if "Churn" in df.columns:
        print("Churn column exists.")
        print(df["Churn"].value_counts(dropna=False))
    else:
        print("ERROR: Churn column is missing.")

    print("\n7. Customer ID validation")
    if "customerID" in df.columns:
        print("Unique customer IDs:", df["customerID"].nunique())
        print("Missing customer IDs:", df["customerID"].isnull().sum())
        print("Duplicate customer IDs:", int(df["customerID"].duplicated().sum()))
    else:
        print("ERROR: customerID column is missing.")

    # Build a column-level validation report
    report_rows = []

    for column in df.columns:
        report_rows.append({
            "Column": column,
            "DataType": str(df[column].dtype),
            "MissingValues": int(df[column].isnull().sum()),
            "MissingPercent": round(
                float(df[column].isnull().mean() * 100), 2
            ),
            "UniqueValues": int(df[column].nunique(dropna=True)),
            "HasMissingValues": bool(df[column].isnull().any())
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\n8. Validation report saved to:")
    print(OUTPUT_PATH)
    print("\nData validation completed successfully.")


if __name__ == "__main__":
    validate_dataset()
