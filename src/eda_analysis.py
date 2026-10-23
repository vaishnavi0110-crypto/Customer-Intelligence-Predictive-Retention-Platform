
from pathlib import Path
import pandas as pd

# 1. Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Locate the cleaned dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def run_eda():
    # 3. Load cleaned customer data
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # 4. Count customers
    total_customers = len(df)
    print("\n1. Total customers:", total_customers)

    # 5. Count churn and non-churn customers
    churn_counts = df["Churn"].value_counts()
    print("\n2. Customer churn counts:")
    print(churn_counts)

    # 6. Calculate churn and retention percentages
    churn_rate = (df["Churn"].eq("Yes").mean()) * 100
    retention_rate = (df["Churn"].eq("No").mean()) * 100

    print(f"\n3. Churn rate: {churn_rate:.2f}%")
    print(f"4. Retention rate: {retention_rate:.2f}%")

    # 7. Compare monthly charges by churn status
    print("\n5. Average monthly charges by churn status:")
    print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

    # 8. Compare customer tenure by churn status
    print("\n6. Average tenure by churn status:")
    print(df.groupby("Churn")["tenure"].mean().round(2))

    # 9. Compare churn across contract types
    print("\n7. Churn counts by contract:")
    print(pd.crosstab(df["Contract"], df["Churn"]))

    # 10. Save a summary report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Churned Customers",
            "Retained Customers",
            "Churn Rate (%)",
            "Retention Rate (%)"
        ],
        "Value": [
            total_customers,
            int(df["Churn"].eq("Yes").sum()),
            int(df["Churn"].eq("No").sum()),
            round(churn_rate, 2),
            round(retention_rate, 2)
        ]
    })

    summary_path = report_dir / "eda_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n8. Summary saved to:", summary_path)


if __name__ == "__main__":
    run_eda()
