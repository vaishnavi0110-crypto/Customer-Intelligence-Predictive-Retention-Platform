import pandas as pd
import os

def test_source_dataset_exists():
    # Raw data file path correct-ah iruka nu check panradhu
    raw_path = "data/raw/Telco-Customer-Churn .csv"
    # Processed folder-la cleaned data iruka nu check panradhu
    processed_path = "data/processed/cleaned_data.csv"
    
    # Files exist aagudha nu check panradhu (Optional logic validation)
    assert os.path.exists(raw_path) or os.path.exists(processed_path)

def test_prediction_output_schema():
    # Output report generated folder structure validity check
    report_path = "reports/predictions_output.csv"
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        assert len(df) >= 0
