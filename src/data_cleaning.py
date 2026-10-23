
from pathlib import Path
import pandas as pd

# 1. Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define input and output paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "telco_churn_cleaned.csv"


def clean_dataset():
    # 3. Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # 4. Remove unwanted spaces from text columns
    text_columns = df.select_dtypes(include=["object", "str"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()

    # 5. Convert TotalCharges into numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # 6. Handle missing TotalCharges
    # The 11 affected customers have tenure = 0.
    zero_tenure_mask = df["tenure"] == 0
    df.loc[
        zero_tenure_mask & df["TotalCharges"].isnull(),
        "TotalCharges"
    ] = 0

    # 7. Report remaining missing values
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # 8. Check duplicate records
    print("\nDuplicate rows:", df.duplicated().sum())

    # 9. Create the processed-data folder if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 10. Save the cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved successfully!")
    print("Output file:", OUTPUT_PATH)
    print("Cleaned dataset shape:", df.shape)


if __name__ == "__main__":
    clean_dataset()
