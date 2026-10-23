# Customer Intelligence & Predictive Retention Platform

## Project Overview

An end-to-end Machine Learning project that analyzes customer behavior, predicts customer churn, segments customers, and generates business insights to support customer retention strategies.

## Business Problem

Customer churn affects recurring revenue and long-term customer relationships. This project uses customer demographic, service, contract, and billing data to identify churn patterns and support data-driven retention decisions.

## Project Features

- Data validation and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Machine learning model comparison
- Customer churn prediction
- K-Means customer segmentation
- Business analytics and churn visualizations
- Streamlit interactive dashboard
- Model evaluation and business insights reports

## Machine Learning Models

The model comparison evaluates:

- Dummy Classifier baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Logistic Regression is the current final candidate based on the observed accuracy and F1-score. Random Forest has higher recall, while Gradient Boosting has the highest ROC-AUC in the recorded comparison. The preferred model depends on the business objective and the cost of missed churn cases.

## Dataset

This project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 original columns.

**Dataset limitations:** The dataset is older and reflects a specific telecommunications business context. Findings may not generalize to current customers or other industries. Observed associations do not establish causation.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Project Structure

```text
Customer_Intelligence_Predictive_Retention_Platform/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── reports/
│   ├── eda_summary.csv
│   ├── model_comparison.csv
│   ├── model_evaluation_report.csv
│   ├── business_insights_report.csv
│   ├── customer_segments.csv
│   ├── customer_segment_summary.csv
│   └── customer_prediction.csv
├── src/
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── verify_cleaned_data.py
│   ├── eda_analysis.py
│   ├── eda_visualizations.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict_customer.py
│   ├── customer_segmentation.py
│   ├── model_evaluation_report.py
│   ├── business_insights_report.py
│   └── dashboard.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Install Python.
2. Clone or download this repository.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Place the dataset in `data/raw/` and run the project's data preparation scripts in the appropriate order.
6. Train the model:

```bash
python src/train.py
```

7. Generate customer segments:

```bash
python src/customer_segmentation.py
```

8. Launch the dashboard:

```bash
python -m streamlit run src/dashboard.py
```

## Current Results

- Dataset size: 7,043 customers
- Observed churn rate: 26.54%
- Logistic Regression accuracy: 80.55%
- Logistic Regression F1-score: 60.40%
- Number of customer segments: 3

These results are from the current evaluation run. They are not a guarantee of performance on new data.

## Future Improvements

- Hyperparameter tuning and cross-validation
- Churn threshold optimization based on business costs
- Model monitoring and drift detection
- Automated tests and reproducible pipelines
- Deployment through a cloud platform

## Disclaimer

This project is intended for educational and portfolio purposes. Predictions are decision-support estimates and should not be treated as certainty about individual customers.