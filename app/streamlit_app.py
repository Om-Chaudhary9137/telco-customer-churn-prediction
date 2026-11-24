"""
streamlit_app.py
Enterprise Retention Intelligence Dashboard connected to FastAPI REST backend.
Supports both:
1. Single Customer Live Simulator + SHAP Waterfall
2. Bulk Customer File Upload (CSV / Excel / Parquet) + Downloadable Scored Report
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RetentionAI | Customer Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_prediction_and_explanation(payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}/explain", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def score_batch_file(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{API_BASE_URL}/predict_batch", files=files, timeout=30)
    response.raise_for_status()
    return response.json()


# Header
st.title("📡 RetentionAI — Customer Churn Intelligence")
st.caption("Real-Time Churn Risk Scoring, Batch Portfolio Analysis & SHAP Explainability")

# Sidebar Status & Single Customer Controls
with st.sidebar:
    st.subheader("System Status")
    if check_api_health():
        st.success("• Online")
    else:
        st.error("• Offline")
        st.warning("Start backend: `uvicorn app.api:app --reload`")
        st.stop()

    st.divider()
    st.subheader("Single Customer Simulator")

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
    tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=5)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure_months * monthly_charges)
    )

    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=0,
    )
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0)
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"], index=0)
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"], index=0)
    partner = st.selectbox("Has Partner?", ["No", "Yes"], index=0)

# Two Main Tabs: Single Analysis vs Batch Upload
tab1, tab2 = st.tabs(["🎯 Single Customer Analysis", "📂 Batch File Upload & Scoring"])

# ==============================================================================
# TAB 1: SINGLE CUSTOMER SIMULATOR
# ==============================================================================
with tab1:
    api_payload = {
        "gender": "Female",
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": tech_support,
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "city": "Los Angeles",
        "zip_code": 90001,
        "latitude": 34.05,
        "longitude": -118.24,
        "cltv": 4000,
    }

    try:
        result = get_prediction_and_explanation(api_payload)
    except Exception as err:
        st.error(f"API Communication Error: {err}")
        st.stop()

    kpi1, kpi2, kpi3 = st.columns(3)
    prob = result["churn_probability"]
    thresh = result["threshold_used"]
    risk_tier = result["risk_tier"]
    eng = result["engineered_summary"]

    with kpi1:
        st.metric("Churn Probability", f"{prob:.1%}", delta=f"{(prob - thresh):+.1%} vs Threshold", delta_color="inverse")
    with kpi2:
        st.metric("Risk Classification", risk_tier, delta=f"Cutoff: {thresh:.1%}", delta_color="off")
    with kpi3:
        st.metric("Monthly Bill Shock", f"${eng['charge_diff_from_avg']:+.2f}", delta=f"Avg: ${eng['avg_monthly_spend']:.2f}/mo", delta_color="inverse")

    st.write("")
    left_col, right_col = st.columns([1, 1.45], gap="large")

    with left_col:
        st.subheader("Recommended Retention Action")
        if risk_tier == "HIGH RISK":
            st.error(f"**Immediate Intervention Required**\n\n{result['recommended_action']}")
        elif risk_tier == "ELEVATED RISK":
            st.warning(f"**Proactive Outreach Recommended**\n\n{result['recommended_action']}")
        else:
            st.success(f"**Account Healthy**\n\n{result['recommended_action']}")

        st.subheader("Behavioral Signal Summary")
        signal_df = pd.DataFrame(
            {
                "Engineered Signal": [
                    "Avg Historical Spend ($/mo)",
                    "Current vs Avg Bill Diff ($)",
                    "Active Add-on Services",
                    "Month-to-Month Flag",
                    "Electronic Check Flag",
                    "Customer Lifecycle Stage",
                ],
                "Value": [
                    f"${eng['avg_monthly_spend']:.2f}",
                    f"${eng['charge_diff_from_avg']:+.2f}",
                    f"{int(eng['total_addon_services'])} / 6",
                    "Yes" if eng["is_month_to_month"] == 1 else "No",
                    "Yes" if eng["has_electronic_check"] == 1 else "No",
                    eng["tenure_group"],
                ],
            }
        )
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Key Risk Drivers (Local SHAP Attribution)")
        st.caption("Red bars increase churn risk; blue bars decrease churn risk.")

        raw_override_map = {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "avg_monthly_spend": eng["avg_monthly_spend"],
            "charge_diff_from_avg": eng["charge_diff_from_avg"],
            "total_addon_services": eng["total_addon_services"],
            "is_month_to_month": eng["is_month_to_month"],
            "has_electronic_check": eng["has_electronic_check"],
        }
        display_values = [
            round(float(raw_override_map[fname]), 2) if fname in raw_override_map else round(float(fval), 2)
            for fname, fval in zip(result["feature_names"], result["feature_values"])
        ]

        explanation = shap.Explanation(
            values=np.array(result["shap_values"]),
            base_values=result["base_value"],
            data=np.array(display_values),
            feature_names=result["feature_names"],
        )

        fig, ax = plt.subplots(figsize=(8, 4.8))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==============================================================================
# TAB 2: BATCH FILE UPLOAD & SCORING
# ==============================================================================
with tab2:
    st.subheader("Bulk Customer Churn Scoring")
    st.markdown(
        "Upload a customer dataset (`.csv`, `.xlsx`, or `.parquet`) to score thousands of customers at once "
        "and export an actionable retention call list."
    )

    uploaded_file = st.file_uploader(
        "Choose a customer file (You can test with `data/processed/test.parquet` or raw `Telco_customer_churn.xlsx`)",
        type=["csv", "xlsx", "parquet"],
    )

    if uploaded_file is not None:
        if st.button("🚀 Run Batch Churn Prediction", type="primary"):
            try:
                with st.spinner(f"Sending '{uploaded_file.name}' to FastAPI for batch scoring..."):
                    batch_response = score_batch_file(uploaded_file)

                total_cust = batch_response["total_customers"]
                churn_count = batch_response["predicted_churners"]
                high_risk = batch_response["high_risk_count"]
                churn_rate = (churn_count / total_cust * 100) if total_cust > 0 else 0.0

                # Batch Summary KPIs
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Customers Scored", f"{total_cust:,}")
                b2.metric("Predicted Churners", f"{churn_count:,}", delta=f"{churn_rate:.1f}% Churn Rate", delta_color="inverse")
                b3.metric("Critical High-Risk (>70%)", f"{high_risk:,}")
                b4.metric("Retained / Safe Customers", f"{(total_cust - churn_count):,}")

                st.write("")
                results_df = pd.DataFrame(batch_response["results"])
                # Filter option for retention team
                filter_option = st.radio(
                    "Filter Results:",
                    ["All Customers", 
                     "Predicted Churners Only (Yes)", 
                     "High Risk Only (>70%)"],
                     horizontal=True,)
                if filter_option == "Predicted Churners Only (Yes)":
                    display_df = results_df[results_df["will_churn"] == "Yes"]
                elif filter_option == "High Risk Only (>70%)":
                    display_df = results_df[results_df["risk_tier"] == "HIGH RISK"]
                else:
                    display_df = results_df
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    # Export Scored CSV Button
                    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored Retention List (.csv)",
                        data=csv_bytes,
                        file_name="scored_churn_retention_list.csv",
                        mime="text/csv",)
            except requests.exceptions.HTTPError as http_err:
                st.error(f"Batch API Error: {http_err.response.text}")
            except Exception as err:
                st.error(f"Unexpected Error: {err}")
st.divider()
st.caption("RetentionAI Enterprise Platform • Powered by FastAPI, LightGBM & SHAP Explainability")
