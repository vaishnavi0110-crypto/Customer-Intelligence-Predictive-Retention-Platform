
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)
SEGMENT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_segment_summary.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_PATH = PROCESSED_DIR / "customer_prediction.csv"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Intelligence & Predictive Retention Platform")
st.write(
    "Analyze customer behavior, explore customer segments, "
    "and predict customer churn."
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run src/train.py first.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# 3A. MODEL COMPARISON
# --------------------------------------------------
st.header("Machine Learning Model Comparison")

MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "notebook_model_evaluation.csv"
)

if MODEL_REPORT_PATH.exists():
    model_results = pd.read_csv(MODEL_REPORT_PATH)

    if not model_results.empty:
        st.dataframe(
            model_results.round(4),
            width="stretch",
            hide_index=True,
        )

        best_f1_row = model_results.loc[
            model_results["F1_Score"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best F1 Model",
            best_f1_row["Model"],
            f"{best_f1_row['F1_Score']:.2%}",
        )

        best_recall_row = model_results.loc[
            model_results["Recall"].idxmax()
        ]
        col2.metric(
            "Best Recall Model",
            best_recall_row["Model"],
            f"{best_recall_row['Recall']:.2%}",
        )

        best_auc_row = model_results.loc[
            model_results["ROC_AUC"].idxmax()
        ]
        col3.metric(
            "Best ROC-AUC Model",
            best_auc_row["Model"],
            f"{best_auc_row['ROC_AUC']:.2%}",
        )

        st.subheader("F1-score Comparison")
        st.bar_chart(
            model_results.set_index("Model")[["F1_Score"]]
        )

else:
    st.warning(
        "Model comparison report not found. "
        "Run python .\\src\\train.py first."
    )

# --------------------------------------------------
# 4. BUSINESS ANALYTICS
# --------------------------------------------------
st.header("Business Analytics")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    required_columns = {"Churn", "Contract"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        st.error(
            "Cleaned dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    else:
        total_customers = len(df)
        churn_customers = int((df["Churn"] == "Yes").sum())
        retained_customers = int((df["Churn"] == "No").sum())

        churn_rate = (
            churn_customers / total_customers * 100
            if total_customers else 0
        )
        retention_rate = (
            retained_customers / total_customers * 100
            if total_customers else 0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churn Customers", f"{churn_customers:,}")
        col3.metric("Churn Rate", f"{churn_rate:.2f}%")
        col4.metric("Retention Rate", f"{retention_rate:.2f}%")

        st.subheader("Customer Churn Distribution")
        churn_counts = df["Churn"].value_counts().reindex(
            ["No", "Yes"], fill_value=0
        )
        st.bar_chart(churn_counts)

        st.subheader("Churn by Contract Type")
        contract_table = pd.crosstab(
            df["Contract"], df["Churn"]
        ).reindex(columns=["No", "Yes"], fill_value=0)
        st.bar_chart(contract_table)

else:
    st.warning(
        "Cleaned dataset not found. Run src/data_cleaning.py first."
    )


# --------------------------------------------------
# 5. CUSTOMER SEGMENTATION
# --------------------------------------------------
st.header("Customer Segmentation")

if SEGMENT_PATH.exists():
    segment_summary = pd.read_csv(SEGMENT_PATH)

    st.write(
        "Customer segment summary generated by the segmentation module."
    )
    st.dataframe(
        segment_summary,
        width="stretch",
        hide_index=True,
    )

    if {"CustomerSegment", "ChurnRatePercent"}.issubset(
        segment_summary.columns
    ):
        st.subheader("Churn Rate by Customer Segment")
        segment_chart = segment_summary.set_index(
            "CustomerSegment"
        )[["ChurnRatePercent"]]
        st.bar_chart(segment_chart)

    elif {"Segment", "ChurnRatePercent"}.issubset(
        segment_summary.columns
    ):
        st.subheader("Churn Rate by Customer Segment")
        segment_chart = segment_summary.set_index(
            "Segment"
        )[["ChurnRatePercent"]]
        st.bar_chart(segment_chart)

else:
    st.warning(
        "Customer segment summary not found at:\n"
        f"{SEGMENT_PATH}\n"
        "Run src/customer_segmentation.py and check its output path."
    )


# --------------------------------------------------
# 6. CUSTOMER CHURN PREDICTION FORM
# --------------------------------------------------
st.header("Predict Customer Churn")
st.write("Enter customer details, then click Predict Churn.")

with st.form("customer_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0, max_value=200.0, value=70.0, step=1.0,
        )
        default_total = float(tenure) * float(monthly_charges)
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0, max_value=10000.0,
            value=min(default_total, 10000.0), step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn", width="stretch"
    )


# --------------------------------------------------
# 7. DISPLAY AND SAVE PREDICTION
# --------------------------------------------------
if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_df = pd.DataFrame([customer_data])

    try:
        prediction = model.predict(customer_df)[0]
        probabilities = model.predict_proba(customer_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            yes_index = classes.index(1)
        elif "Yes" in classes:
            yes_index = classes.index("Yes")
        else:
            raise ValueError(
                f"Cannot find the churn-positive class in {classes}"
            )

        churn_probability = float(probabilities[yes_index]) * 100
        predicted_yes = prediction == 1 or prediction == "Yes"
        predicted_label = "Yes" if predicted_yes else "No"

        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if predicted_yes:
                st.error("Prediction: Customer may churn")
            else:
                st.success("Prediction: Customer likely to stay")

        with result_col2:
            st.metric("Churn Probability", f"{churn_probability:.2f}%")

        st.progress(min(max(churn_probability / 100, 0.0), 1.0))

        if churn_probability >= 50:
            st.warning(
                "Higher predicted churn risk. Consider suitable "
                "customer-retention actions."
            )
        else:
            st.info(
                "Lower predicted churn risk. Continue monitoring "
                "customer satisfaction."
            )

        prediction_report = customer_df.copy()
        prediction_report["PredictedChurn"] = predicted_label
        prediction_report["ChurnProbability"] = round(
            churn_probability, 2
        )
        prediction_report.to_csv(PREDICTION_PATH, index=False)

        st.success(f"Prediction saved to: {PREDICTION_PATH}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Customer Intelligence & Predictive Retention Platform | "
    "Machine Learning Internship Project"
)
