# Customer Intelligence & Predictive Retention Platform

An end-to-end customer intelligence and predictive churn retention prototype built during the AI Data Analytics Internship. The platform combines machine learning classification, customer segmentation, an interactive decision-support dashboard, and a production-ready REST API endpoint.

## 🚀 Technology Focus
* **Language:** Python 3.13
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, Joblib
* **User Interface:** Streamlit Dashboard
* **REST API:** FastAPI & Uvicorn
* **Testing:** Pytest

## 📁 Repository Structure
```text
customer-intelligence-platform/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── segmentation.py
│   └── recommendations.py
├── tests/
│   └── test_kpi.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model.pkl
│   └── kmeans_model.pkl
├── reports/
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Execution Instructions

### 1. Environment Installation
Install all required libraries using the local terminal environment:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Testing Suite
Verify code quality, dataset validation checks, and KPI calculations before execution:
```bash
pytest
```

### 3. Launch FastAPI Predictive Service
Start the backend local server endpoint for predictive model scoring:
```bash
uvicorn api.main:app --reload
```
* Access Swagger API documentation at: `http://127.0.0`

### 4. Launch Streamlit Analytics Dashboard
Run the front-end interactive workspace application:
```bash
streamlit run app/streamlit_app.py
```

## 📊 Core Features & Key Metrics
* **Classification Pipeline:** Compares ML architectures (Logistic Regression, Random Forest, Gradient Boosting) to compute structural churn risk categories.
* **Customer Segmentation:** Implements K-Means cluster grouping to analyze behavior profiles and customer value classifications.
* **Automated Rules Engine:** Formulates rule-based retention decision suggestions for high-risk accounts.
* **KPI Workspace:** Live calculation matrices mapping total revenue exposure, customer volume counts, and retention rates.
