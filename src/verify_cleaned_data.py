
from pathlib import Path
import pandas as pd

# 1. Find the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Locate the cleaned dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def verify_cleaned_data():
    # 3. Load the cleaned CSV
    df = pd.read_csv(DATA_PATH)

    print("CLEANED DATASET VERIFICATION")
    print("-" * 40)

    # 4. Check the number of rows and columns
    print("Dataset shape:", df.shape)

    # 5. Verify the TotalCharges data type
    print("\nTotalCharges data type:")
    print(df["TotalCharges"].dtype)

    # 6. Verify missing values
    print("\nTotal missing values:")
    print(df.isnull().sum().sum())

    # 7. Verify churn counts
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    # 8. Show the first five records
    print("\nFirst five rows:")
    print(df.head())

    # 9. Check expected dataset structure
    print("\nExpected structure correct:",
          df.shape == (7043, 21))

    print("\nVerification completed.")


if __name__ == "__main__":
    verify_cleaned_data()
