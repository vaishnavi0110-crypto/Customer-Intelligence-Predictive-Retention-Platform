# Customer Intelligence & Predictive Retention Platform

An end-to-end customer intelligence and predictive churn retention prototype developed during an AI Data Analytics Internship. This project combines customer data preparation, machine learning model evaluation, customer segmentation, an interactive Streamlit dashboard, and a FastAPI prediction service.

## Technology Stack

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Dashboard:** Streamlit
- **REST API:** FastAPI, Uvicorn
- **Testing:** Pytest
- **Dataset:** IBM Telco Customer Churn Dataset (7,043 customer records)

## Repository Structure

```text
Customer_Intelligence_Predictive_Retention_Platform/
├── api/
│   ├── __init__.py
│   └── main.py
├── app/
│   ├── __init__.py
│   └── main.py
├── src/
│   ├── business_insights_report.py
│   ├── customer_segmentation.py
│   ├── dashboard.py
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── eda_analysis.py
│   ├── eda_visualizations.py
│   ├── feature_engineering.py
│   ├── model_evaluation_report.py
│   ├── predict_customer.py
│   ├── preprocessing.py
│   ├── train.py
│   └── verify_cleaned_data.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Model_Evaluation.ipynb
│   └── 03_Customer_Segmentation.ipynb
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
├── models/
│   └── churn_model.joblib
├── reports/
│   ├── confusion_matrix.png
│   ├── notebook_churn_distribution.png
│   ├── notebook_churn_summary.csv
│   ├── notebook_contract_churn.csv
│   ├── notebook_contract_churn.png
│   ├── notebook_customer_summary.csv
│   ├── notebook_model_evaluation.csv
│   ├── notebook_monthly_charges_churn.png
│   ├── notebook_tenure_churn.png
│   └── roc_curve.png
├── requirements.txt
├── .gitignore
└── README.md
```
*Note: The structure above highlights the main project components. Refer to the repository for the complete list of files.*

## Setup and Execution

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Launch the Streamlit dashboard

```powershell
python -m streamlit run .\src\dashboard.py
```

Open `http://localhost:8501` in your browser.

### 4. Launch the FastAPI service

Open a separate terminal in the project directory and activate the virtual environment, then run:

```powershell
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore and test the API endpoints.

### 5. Run automated tests

```powershell
python -m pytest -v
```

## Core Features

### Machine Learning Model Comparison

Compares classification models, including Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, and Dummy Classifier using evaluation metrics such as F1-score, Recall, and ROC-AUC.

### Customer Analytics

Provides customer counts, churn and retention rates, churn distribution, and churn patterns by contract type.

### Customer Segmentation

Displays customer segment summaries and churn-rate comparisons across segments.

### Churn Prediction

Allows users to enter customer details and receive a churn prediction with an estimated probability.

### REST API

Provides a FastAPI service for submitting customer details and receiving prediction results. Interactive API documentation is available through Swagger UI.

## Current Dashboard Metrics

The dashboard currently reports:

- **Total customers:** 7,043
- **Churn customers:** 1,869
- **Churn rate:** 26.54%
- **Retention rate:** 73.46%
- **Best F1 model:** Logistic Regression — 60.40%
- **Best Recall model:** Logistic Regression — 55.88%
- **Best ROC-AUC model:** Gradient Boosting — 84.33%

These are the results recorded in `reports/notebook_model_evaluation.csv`. Model performance may vary when retraining or changing the evaluation setup.

These are the results currently displayed by the project dashboard; model performance may vary when retraining or changing the evaluation setup.

## Data and Model Notes

- The dataset contains historical customer information and may not represent current customer behavior.
- Customer data, processed datasets, and trained model artifacts may be excluded from version control by `.gitignore`.
- Reproduce the dashboard and API outputs only after obtaining the required local dataset and model files.
- Predictions are estimates for decision support, not guarantees of future customer behavior.

### Dataset Documentation

- **Dataset Name:** IBM Telco Customer Churn Dataset
- **Dataset Source:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **Expected Records:** 7,043
- **Expected Columns:** 21
- **Customer Identifier:** `customerID`
- **Target Column:** `Churn`

Verify the record and column counts against the actual CSV file before reporting them as confirmed.

### Dataset Limitations

- The dataset contains historical information and may not reflect current customer behavior.
- It is not a live feed from a company's CRM or billing system.
- Support tickets, complaints, customer satisfaction scores, or usage information must not be presented as real observations unless those fields exist in the source data.
- Model predictions are estimates and should support, not replace, business decisions.

### Data Quality and Preprocessing

- Check for missing values and duplicate records.
- Validate column names and data types.
- Convert `TotalCharges` to numeric and inspect blank or invalid values.
- Document the cleaning, preprocessing, and feature engineering steps performed before model training.

### Reproducibility and Responsible Use

- Record the dataset source and any modifications made to the original data.
- Keep credentials and unauthorized private customer information out of the repository.
- Ensure that the required dataset and trained model files are available before running the dashboard or API.