"""
api.py
Production-grade FastAPI REST microservice supporting:
1. Single-customer real-time scoring (/predict)
2. Local SHAP explainability (/explain)
3. Bulk file upload batch scoring (/predict_batch) for CSV, Excel, and Parquet files.
"""

import io
import traceback
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.config import MODEL_PIPELINE_PATH
from src.data_prep import to_snake_case

app = FastAPI(
    title="Telco Customer Churn Prediction & Explainability API",
    description="Decoupled ML microservice serving real-time & batch churn scoring.",
    version="2.1.0",
)

OPTIMAL_THRESHOLD = 0.535

# Load pipeline and SHAP explainer ONCE at startup
try:
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
except Exception as e:
    pipeline = None
    explainer = None
    print(f"CRITICAL ERROR: Failed to load model pipeline: {e}")


class CustomerFeatures(BaseModel):
    """Pydantic schema validating single incoming JSON payloads."""
    gender: str = Field(..., example="Female")
    senior_citizen: str = Field(..., example="No")
    partner: str = Field(..., example="No")
    dependents: str = Field(..., example="No")
    tenure_months: int = Field(..., ge=0, le=120, example=5)
    phone_service: str = Field(..., example="Yes")
    multiple_lines: str = Field(..., example="No")
    internet_service: str = Field(..., example="Fiber optic")
    online_security: str = Field(..., example="No")
    online_backup: str = Field(..., example="No")
    device_protection: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    streaming_tv: str = Field(..., example="Yes")
    streaming_movies: str = Field(..., example="No")
    contract: str = Field(..., example="Month-to-month")
    paperless_billing: str = Field(..., example="Yes")
    payment_method: str = Field(..., example="Electronic check")
    monthly_charges: float = Field(..., ge=0.0, example=85.0)
    total_charges: float = Field(..., ge=0.0, example=425.0)
    city: str = Field(default="Los Angeles", example="Los Angeles")
    zip_code: int = Field(default=90001, example=90001)
    latitude: float = Field(default=34.05, example=34.05)
    longitude: float = Field(default=-118.24, example=-118.24)
    cltv: int = Field(default=4000, example=4000)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    threshold_used: float
    recommended_action: str


class ExplanationResponse(PredictionResponse):
    base_value: float
    shap_values: list[float]
    feature_names: list[str]
    feature_values: list[float]
    engineered_summary: dict


def _assign_risk_and_action(prob: float) -> tuple[int, str, str]:
    """Maps continuous probability to binary prediction, risk tier, and retention action."""
    pred = int(prob >= OPTIMAL_THRESHOLD)
    if prob >= 0.70:
        return pred, "HIGH RISK", "Immediate retention call + 15% 1-Year contract discount."
    elif pred == 1:
        return pred, "ELEVATED RISK", "Free 3-month Online Security & Tech Support + Auto-Pay offer."
    else:
        return pred, "LOW RISK", "No discount needed; maintain standard engagement."


def _score_customer(input_df: pd.DataFrame) -> dict:
    # Extract class-1 probability cleanly as a Python float
    probs_matrix = pipeline.predict_proba(input_df)
    prob = float(probs_matrix[:, 1].item())
    pred, risk_tier, action = _assign_risk_and_action(prob)
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "risk_tier": risk_tier,
        "threshold_used": OPTIMAL_THRESHOLD,
        "recommended_action": action,
    }


@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    return {"status": "healthy", "model_version": "2.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable.")
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        return PredictionResponse(**_score_customer(input_df))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
def explain_churn(customer: CustomerFeatures):
    if pipeline is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model or explainer unavailable.")
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        base_result = _score_customer(input_df)

        # 1. Extract first row of engineered features safely as a dictionary
        engineered_df = preprocessor.named_steps["feature_engineering"].transform(input_df)
        first_row = engineered_df.to_dict(orient="records").pop()
        eng_cols = [
            "avg_monthly_spend",
            "charge_diff_from_avg",
            "total_addon_services",
            "is_month_to_month",
            "has_electronic_check",
            "tenure_group",
        ]
        engineered_summary = {}
        for col in eng_cols:
            val = first_row[col]
            if isinstance(val, (int, float, np.number)):
                engineered_summary[col] = round(float(val), 2)
            else:
                engineered_summary[col] = str(val)

        # 2. Compute transformed matrix & local SHAP values
        X_trans = preprocessor.transform(input_df)
        feature_names = preprocessor.named_steps["preprocessing"].get_feature_names_out().tolist()
        shap_exp = explainer(X_trans)

        vals = shap_exp.values
        if vals.ndim == 3:
            shap_vals = vals[:, :, 1].flatten().tolist()
        else:
            shap_vals = vals.flatten().tolist()

        base_val = float(np.ravel(shap_exp.base_values)[-1])
        feat_vals = np.ravel(X_trans).tolist()

        return ExplanationResponse(
            **base_result,
            base_value=base_val,
            shap_values=shap_vals,
            feature_names=feature_names,
            feature_values=feat_vals,
            engineered_summary=engineered_summary,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV, Excel (.xlsx), or Parquet file containing multiple customers,
    cleans schema automatically, runs vectorized batch inference, and returns scored records.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable.")

    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload .csv, .xlsx, or .parquet",
            )

        df.columns = [to_snake_case(col) for col in df.columns]

        customer_ids = (
            df["customerid"].astype(str).tolist()
            if "customerid" in df.columns
            else [f"CUST-{i+1:04d}" for i in range(len(df))]
        )

        if "total_charges" in df.columns:
            df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)

        defaults = {
            "city": "Los Angeles",
            "zip_code": 90001,
            "latitude": 34.05,
            "longitude": -118.24,
            "cltv": 4000,
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        probs = pipeline.predict_proba(df)[:, 1]
        records_list = df.to_dict(orient="records")

        scored_records = []
        for idx, prob in enumerate(probs):
            prob_float = float(prob)
            pred, risk_tier, action = _assign_risk_and_action(prob_float)
            row_dict = records_list[idx]
            scored_records.append(
                {
                    "customer_id": customer_ids[idx],
                    "contract": str(row_dict.get("contract", "N/A")),
                    "tenure_months": int(row_dict.get("tenure_months", 0)),
                    "monthly_charges": round(float(row_dict.get("monthly_charges", 0.0)), 2),
                    "churn_probability": round(prob_float, 4),
                    "will_churn": "Yes" if pred == 1 else "No",
                    "risk_tier": risk_tier,
                    "recommended_action": action,
                }
            )

        return {
            "total_customers": len(scored_records),
            "predicted_churners": sum(1 for r in scored_records if r["will_churn"] == "Yes"),
            "high_risk_count": sum(1 for r in scored_records if r["risk_tier"] == "HIGH RISK"),
            "results": scored_records,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")
