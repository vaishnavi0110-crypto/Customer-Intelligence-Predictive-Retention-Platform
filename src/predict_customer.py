
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Load the saved preprocessing + prediction pipeline
model = joblib.load(MODEL_PATH)

# Sample customer details
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.50,
    "TotalCharges": 450.00
}])

# Predict churn: 1 = Yes, 0 = No
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]


result = customer.copy()
result["PredictedChurn"] = "Yes" if prediction == 1 else "No"
result["ChurnProbability"] = round(probability, 4)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "customer_prediction.csv"
result.to_csv(OUTPUT_PATH, index=False)

print("\nCUSTOMER CHURN PREDICTION")
print("-------------------------")
print("Predicted churn:", result["PredictedChurn"].iloc[0])
print(f"Estimated churn probability: {probability:.2%}")
print("Prediction report saved to:", OUTPUT_PATH)
