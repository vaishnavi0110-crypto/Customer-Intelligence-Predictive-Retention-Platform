
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def run_segmentation():
    # 1. Load cleaned customer data
    df = pd.read_csv(DATA_PATH)

    # 2. Select useful customer behavior features
    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    X = df[features].copy()

    # 3. Scale features so their magnitudes are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Group customers into 3 clusters
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    df["CustomerSegment"] = kmeans.fit_predict(X_scaled)

    # 5. Create a summary for each segment
    summary = (
        df.groupby("CustomerSegment")
        .agg(
            CustomerCount=("customerID", "count"),
            AverageTenure=("tenure", "mean"),
            AverageMonthlyCharges=("MonthlyCharges", "mean"),
            AverageTotalCharges=("TotalCharges", "mean"),
            ChurnCount=("Churn", lambda values: (values == "Yes").sum()),
        )
        .reset_index()
    )

    summary["ChurnRatePercent"] = (
        summary["ChurnCount"] / summary["CustomerCount"] * 100
    ).round(2)

    # 6. Save results
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    segment_path = REPORT_DIR / "customer_segments.csv"
    summary_path = REPORT_DIR / "customer_segment_summary.csv"

    df.to_csv(segment_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nCUSTOMER SEGMENTATION COMPLETED")
    print("Total customers:", len(df))
    print("Number of segments:", df["CustomerSegment"].nunique())

    print("\nSEGMENT SUMMARY")
    print(summary.round(2).to_string(index=False))

    print("\nSaved customer segments to:", segment_path)
    print("Saved segment summary to:", summary_path)


if __name__ == "__main__":
    run_segmentation()
