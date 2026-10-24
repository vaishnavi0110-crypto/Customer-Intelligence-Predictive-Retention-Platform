
from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cleaned dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

# Output directory
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Verification report path
OUTPUT_PATH = PROCESSED_DIR / "cleaned_data_verification.csv"


def verify_cleaned_data():
    # Check whether cleaned dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run src/data_cleaning.py first."
        )

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("CLEANED DATASET VERIFICATION")
    print("=" * 50)

    # Dataset dimensions
    print("\n1. Dataset shape:")
    print(df.shape)

    # Required columns
    required_columns = [
        "customerID",
        "TotalCharges",
        "Churn"
    ]

    print("\n2. Required column checks:")
    for column in required_columns:
        print(f"{column}: {'Found' if column in df.columns else 'MISSING'}")

    # TotalCharges data type
    if "TotalCharges" in df.columns:
        print("\n3. TotalCharges data type:")
        print(df["TotalCharges"].dtype)

    # Missing values
    print("\n4. Missing values by column:")
    print(df.isnull().sum())
    total_missing = int(df.isnull().sum().sum())
    print("Total missing values:", total_missing)

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    print("\n5. Duplicate rows:", duplicate_rows)

    # Churn distribution
    if "Churn" in df.columns:
        print("\n6. Churn distribution:")
        print(df["Churn"].value_counts(dropna=False))

    # Customer ID checks
    if "customerID" in df.columns:
        print("\n7. Customer ID checks:")
        print("Missing IDs:", int(df["customerID"].isnull().sum()))
        print("Unique IDs:", int(df["customerID"].nunique()))

    # Dataset structure check
    expected_shape = (7043, 21)
    structure_correct = df.shape == expected_shape

    print("\n8. Expected shape:", expected_shape)
    print("Expected structure correct:", structure_correct)

    # Create verification report
    report = pd.DataFrame({
        "Check": [
            "Rows",
            "Columns",
            "Total missing values",
            "Duplicate rows",
            "Expected shape (7043, 21)",
            "Has customerID column",
            "Has TotalCharges column",
            "Has Churn column"
        ],
        "Result": [
            df.shape[0],
            df.shape[1],
            total_missing,
            duplicate_rows,
            structure_correct,
            "customerID" in df.columns,
            "TotalCharges" in df.columns,
            "Churn" in df.columns
        ]
    })

    report.to_csv(OUTPUT_PATH, index=False)

    print("\nVerification report saved to:")
    print(OUTPUT_PATH)
    print("\nVerification completed successfully.")


if __name__ == "__main__":
    verify_cleaned_data()
