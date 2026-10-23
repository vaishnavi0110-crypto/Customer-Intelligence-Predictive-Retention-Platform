
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_evaluation_report.csv"


def main():
    if not INPUT_PATH.exists():
        print("ERROR: reports/model_comparison.csv not found.")
        return

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "Model", "Accuracy", "Precision",
        "Recall", "F1", "ROC_AUC"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print("Missing columns:", missing)
        print("Available columns:", df.columns.tolist())
        return

    best_accuracy = df.loc[df["Accuracy"].idxmax(), "Model"]
    best_recall = df.loc[df["Recall"].idxmax(), "Model"]
    best_f1 = df.loc[df["F1"].idxmax(), "Model"]
    best_roc_auc = df.loc[df["ROC_AUC"].idxmax(), "Model"]

    report = df.copy()
    report["Best_Accuracy"] = report["Model"].eq(best_accuracy)
    report["Best_Recall"] = report["Model"].eq(best_recall)
    report["Best_F1"] = report["Model"].eq(best_f1)
    report["Best_ROC_AUC"] = report["Model"].eq(best_roc_auc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nMODEL EVALUATION REPORT")
    print("-----------------------")
    print(report.to_string(index=False))
    print("\nBest Accuracy:", best_accuracy)
    print("Best Recall:", best_recall)
    print("Best F1-score:", best_f1)
    print("Best ROC-AUC:", best_roc_auc)
    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
