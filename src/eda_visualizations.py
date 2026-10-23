
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Load the cleaned dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

# 3. Set the folder for saving graphs
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def create_churn_graphs():
    df = pd.read_csv(DATA_PATH)

    # GRAPH 1: Churn vs Retention
    churn_counts = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    plt.figure(figsize=(7, 5))
    churn_counts.plot(kind="bar")
    plt.title("Customer Churn vs Retention")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()

    graph1_path = REPORT_DIR / "churn_distribution.png"
    plt.savefig(graph1_path, dpi=150)
    plt.close()

    # GRAPH 2: Churn by Contract Type
    contract_counts = pd.crosstab(
        df["Contract"], df["Churn"]
    ).reindex(columns=["No", "Yes"], fill_value=0)

    contract_counts.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Customer Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()

    graph2_path = REPORT_DIR / "churn_by_contract.png"
    plt.savefig(graph2_path, dpi=150)
    plt.close()

    print("Graphs created successfully!")
    print("Graph 1:", graph1_path)
    print("Graph 2:", graph2_path)


if __name__ == "__main__":
    create_churn_graphs()
