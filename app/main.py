
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_FILE = PROJECT_ROOT / "src" / "dashboard.py"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Intelligence & Predictive Retention Platform")

st.write(
    "Analyze customer churn, evaluate machine learning models, "
    "and explore customer segments."
)

st.subheader("Project Dashboard")

if DASHBOARD_FILE.exists():
    st.success("Existing project dashboard found.")
    st.caption(f"Dashboard file: {DASHBOARD_FILE}")
    st.info(
        "To launch your existing dashboard, run "
        "`python -m streamlit run src/dashboard.py` "
        "from the project root folder."
    )
else:
    st.warning(
        "The existing dashboard was not found at src/dashboard.py. "
        "Check your project folder structure."
    )
