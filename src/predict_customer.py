
from pathlib import Path

import joblib
import pandas as pd

# Project folder path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Saved machine learning model
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Output folder
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the trained model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Please run src/train.py first."
    )

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

# Prepare prediction result
result = customer.copy()
result["PredictedChurn"] = "Yes" if prediction == 1 else "No"
result["ChurnProbability"] = round(float(probability), 4)

# Save prediction CSV inside data/processed
OUTPUT_PATH = PROCESSED_DIR / "customer_prediction.csv"
result.to_csv(OUTPUT_PATH, index=False)

# Display results
print("\nCUSTOMER CHURN PREDICTION")
print("-------------------------")
print("Predicted churn:", result["PredictedChurn"].iloc[0])
print(f"Estimated churn probability: {probability:.2%}")
print("Prediction report saved to:", OUTPUT_PATH)
