
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_segmentation():
    # Load cleaned customer data
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Select customer behavior features
    features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X = df[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create 3 customer segments
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )
    df["CustomerSegment"] = kmeans.fit_predict(X_scaled)

    # Create segment summary
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

    # Save BOTH CSV files only in data/processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    segment_path = PROCESSED_DIR / "customer_segments.csv"
    summary_path = PROCESSED_DIR / "customer_segment_summary.csv"

    df.to_csv(segment_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Display results
    print("\nCUSTOMER SEGMENTATION COMPLETED")
    print("Total customers:", len(df))
    print("Number of segments:", df["CustomerSegment"].nunique())

    print("\nSEGMENT SUMMARY")
    print(summary.round(2).to_string(index=False))

    print("\nSaved customer segments to:", segment_path)
    print("Saved segment summary to:", summary_path)


if __name__ == "__main__":
    run_segmentation()
