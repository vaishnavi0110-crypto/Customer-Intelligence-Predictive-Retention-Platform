
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)
OUTPUT_PATH = (PROJECT_ROOT / "data" / "processed" / "business_insights_report.csv")


def main():
    if not DATA_PATH.exists():
        print("ERROR: Cleaned dataset not found.")
        return

    df = pd.read_csv(DATA_PATH)

    total_customers = len(df)
    churn_customers = int((df["Churn"] == "Yes").sum())
    churn_rate = (
        churn_customers / total_customers * 100
        if total_customers else 0
    )

    insights = [
        {
            "Insight": "Total Customers",
            "Finding": total_customers,
            "Business_Action": "Use this as the customer base for retention planning."
        },
        {
            "Insight": "Customers Who Churned",
            "Finding": churn_customers,
            "Business_Action": "Investigate churn reasons and retention opportunities."
        },
        {
            "Insight": "Overall Churn Rate (%)",
            "Finding": round(churn_rate, 2),
            "Business_Action": "Track this metric over time to evaluate retention efforts."
        },
    ]

    # Churn rate by contract type
    contract_summary = (
        df.groupby("Contract")
        .agg(
            TotalCustomers=("Churn", "size"),
            ChurnCustomers=("Churn", lambda x: (x == "Yes").sum()),
        )
        .reset_index()
    )

    contract_summary["ChurnRatePercent"] = (
        contract_summary["ChurnCustomers"]
        / contract_summary["TotalCustomers"] * 100
    ).round(2)

    highest_churn_contract = contract_summary.loc[
        contract_summary["ChurnRatePercent"].idxmax()
    ]

    insights.append({
        "Insight": "Highest Churn Contract Type",
        "Finding": (
            f"{highest_churn_contract['Contract']} "
            f"({highest_churn_contract['ChurnRatePercent']:.2f}%)"
        ),
        "Business_Action": (
            "Review the customer experience and retention offers "
            "for this contract group."
        )
    })

    # Average monthly charges by churn status
    monthly_summary = df.groupby("Churn")["MonthlyCharges"].mean()

    if "Yes" in monthly_summary.index and "No" in monthly_summary.index:
        insights.append({
            "Insight": "Average Monthly Charges",
            "Finding": (
                f"Churned: ${monthly_summary['Yes']:.2f}; "
                f"Retained: ${monthly_summary['No']:.2f}"
            ),
            "Business_Action": (
                "Explore whether pricing, service usage, or customer needs "
                "are associated with churn."
            )
        })

    report = pd.DataFrame(insights)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nBUSINESS INSIGHTS REPORT")
    print("------------------------")
    print(report.to_string(index=False))
    print("\nContract summary:")
    print(contract_summary.to_string(index=False))
    print("\nSaved report to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
