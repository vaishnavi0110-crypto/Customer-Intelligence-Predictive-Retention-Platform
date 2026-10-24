# Customer Intelligence & Predictive Retention Platform

## 1. Project Overview

An end-to-end Machine Learning project that analyzes customer behavior, predicts customer churn, segments customers, and generates business insights to support customer retention strategies.

The project uses the IBM Telco Customer Churn dataset and provides an interactive Streamlit dashboard for exploring business analytics, comparing machine learning models, viewing customer segments, and predicting churn risk.

## 2. Business Problem

Customer churn can reduce recurring revenue and weaken long-term customer relationships. This project analyzes customer demographics, service subscriptions, contract types, tenure, and billing information to identify churn patterns and support data-driven retention decisions.

## 3. Project Features

- Dataset validation and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Machine learning model training and comparison
- Customer churn prediction
- K-Means customer segmentation
- Churn and retention KPI analysis
- Churn distribution and contract analysis
- Interactive Streamlit dashboard
- Model evaluation and business insights reports
- Saved model pipeline for predictions

## 4. Machine Learning Models

Five classification models were trained and evaluated using a stratified 80/20 train-test split with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.21% |
| Random Forest | 77.00% | 55.90% | 63.37% | 59.40% | 82.25% |
| Gradient Boosting | 80.27% | 66.55% | 51.60% | 58.13% | 84.33% |
| Decision Tree | 74.31% | 51.68% | 49.47% | 50.55% | 66.40% |
| Dummy Classifier | 73.46% | 0.00% | 0.00% | 0.00% | 50.00% |

### Model Selection

- **Logistic Regression:** Selected as the current final model because it achieved the highest F1-score (60.40%) in this evaluation.
- **Random Forest:** Achieved the highest recall (63.37%), making it a useful candidate when identifying more potential churners is important.
- **Gradient Boosting:** Achieved the highest ROC-AUC (84.33%) in this evaluation.
- **Dummy Classifier:** Serves as the baseline. Its accuracy alone is misleading because its recall and F1-score are both zero.

The final saved model is selected using the highest test F1-score. The preferred model may change depending on business priorities, the cost of missed churners, and the cost of retention offers.

These results represent one holdout evaluation and are not a guarantee of performance on future customers.

## 5. Dataset

This project uses the IBM Telco Customer Churn dataset.

- Original records: 7,043 customers
- Original columns: 21
- Target variable: `Churn`
- Target classes: `Yes` and `No`

### Dataset Limitations

- The dataset is older and reflects a specific telecommunications business context.
- Results may not generalize to current customers or other industries.
- Observed relationships do not establish causation.
- Predictions should be treated as decision-support estimates, not certainties.

## 6. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Jupyter Notebook

## 7. Project Structure

```text
Customer_Intelligence_Predictive_Retention_Platform/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Model_Evaluation.ipynb
│   └── 03_Customer_Segmentation.ipynb
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
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure above lists the main project files. Your local folders may contain additional generated reports or datasets.*

## 8. Setup Instructions

### Prerequisites

- Python installed
- Git (optional, for cloning the repository)
- Visual Studio Code (recommended)

### Install Dependencies

Open a terminal in the project root directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Prepare the Dataset

Place the original IBM Telco Customer Churn CSV in `data/raw/`.

Run the data preparation scripts in the order required by the project. Ensure the cleaned dataset is created at:

```text
data/processed/telco_churn_cleaned.csv
```

### Train and Compare Models

```bash
python src/train.py
```

This trains and evaluates the five models, saves their comparison results to `reports/notebook_model_evaluation.csv`, and saves the selected model to `models/churn_model.joblib`.

### Generate the Model Evaluation Report

```bash
python src/model_evaluation_report.py
```

The report is saved to `data/processed/model_evaluation_report.csv`.

### Generate Customer Segments

```bash
python src/customer_segmentation.py
```

### Launch the Dashboard

```bash
python -m streamlit run src/dashboard.py
```

Open the local URL displayed in the terminal to use the dashboard.

## 9. Current Results

- Dataset size: 7,043 customers
- Observed churn rate: approximately 26.54%
- Logistic Regression accuracy: 80.55%
- Logistic Regression F1-score: 60.40%
- Random Forest recall: 63.37%
- Gradient Boosting ROC-AUC: 84.33%
- Customer segments: 3 in the current segmentation run

Results may vary if the dataset, preprocessing, split, or model settings change.

## 10. Future Improvements

- Hyperparameter tuning and cross-validation
- Churn threshold optimization based on business costs
- Model calibration and probability validation
- Automated testing and reproducible pipelines
- Model monitoring and data-drift detection
- Dashboard deployment to a cloud platform
- Additional retention-strategy evaluation

## 11. Disclaimer

This project is intended for educational, internship, and portfolio purposes. Model predictions are estimates based on historical data and should not be treated as certainty about individual customers. Business decisions should consider context, fairness, and the consequences of incorrect predictions.