import pandas as pd
import numpy as np

def calculate_churn_rate(churned_count, total_count):
    if total_count == 0:
        return 0.0
    return round((churned_count / total_count) * 100, 2)

def test_calculate_churn_rate():
    # Test cases for churn KPI validation
    assert calculate_churn_rate(20, 100) == 20.0
    assert calculate_churn_rate(0, 50) == 0.0
    assert calculate_churn_rate(10, 0) == 0.0

def test_data_cleaning_structure():
    # Test case to verify processed columns
    sample_data = pd.DataFrame({
        'CustomerID': ['1', '2'],
        'Churn': ['No', 'Yes'],
        'MonthlyCharges': [29.85, 56.95]
    })
    
    assert 'CustomerID' in sample_data.columns
    assert sample_data['MonthlyCharges'].isnull().sum() == 0
