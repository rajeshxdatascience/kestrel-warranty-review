
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models" / "fraud_model_rf.joblib")
partners = pd.read_csv(ROOT / "data" / "partners.csv")
products = pd.read_csv(ROOT / "data" / "products.csv")

app = FastAPI(title="Kestrel Warranty Risk API")


class Claim(BaseModel):
    claim_id: str
    submitted_at: str
    partner_id: str
    sku: str
    product_serial: str
    days_since_purchase: int = Field(ge=0)
    claim_amount_inr: float = Field(gt=0)
    photo_attached: str
    partner_inspected: str
    claim_description: str
    customer_prior_claims: int = Field(ge=0)
    source: str


def prepare_features(record):
    x = pd.DataFrame([record])

    x["submitted_at"] = pd.to_datetime(
        x["submitted_at"], errors="coerce"
    )

    x = x.merge(partners, on="partner_id", how="left")
    x = x.merge(products, on="sku", how="left")

    if pd.isna(x["city"].iloc[0]) or pd.isna(x["family"].iloc[0]):
        raise ValueError("Unknown partner_id or SKU.")

    x["submitted_month"] = (
        x["submitted_at"].dt.month.astype("Int64").astype(str)
    )
    x["submitted_year"] = (
        x["submitted_at"].dt.year.astype("Int64").astype(str)
    )
    x["submitted_weekday"] = (
        x["submitted_at"].dt.dayofweek.astype("Int64").astype(str)
    )

    x["post_policy_change"] = (
        x["submitted_at"] >= pd.Timestamp("2026-05-01")
    ).astype(int)

    x["partner_tenure_days"] = (
        x["submitted_at"] -
        pd.to_datetime(x["onboarded_date"], errors="coerce")
    ).dt.days

    serial = (
        x["product_serial"].fillna("").astype(str)
        .str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    x["serial_length"] = serial.str.len()
    x["serial_has_letters"] = serial.str.contains(r"[A-Z]").astype(int)
    x["serial_has_digits"] = serial.str.contains(r"\d").astype(int)

    x["description_text"] = (
        x["claim_description"].fillna("").astype(str).str.lower()
    )
    x["description_length"] = x["description_text"].str.len()
    x["description_digit_count"] = x["description_text"].str.count(r"\d")

    return x


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(claim: Claim):
    try:
        record = claim.model_dump()
        features = prepare_features(record)
        score = float(model.predict_proba(features)[0, 1])

        reasons = []

        # Explain observable claim characteristics, not guilt.
        if record["claim_amount_inr"] < 2000 and record["submitted_at"] >= "2026-05-01":
            reasons.append(
                "Claim is below Rs 2,000 after the May 2026 policy change; "
                "missing inspection may be permitted by policy."
            )

        if record["customer_prior_claims"] >= 3:
            reasons.append(
                "Customer has multiple prior claims; check claim history."
            )

        if record["days_since_purchase"] <= 30:
            reasons.append(
                "Claim was submitted within 30 days of purchase; "
                "verify purchase and warranty details."
            )

        if not reasons:
            reasons.append(
                "No single rule-based concern identified; review the claim "
                "details and score together."
            )

        return {
            "claim_id": record["claim_id"],
            "risk_score": round(score, 6),
            "interpretation": (
                "Higher score means higher model-estimated fraud risk. "
                "This is a review signal, not proof of fraud."
            ),
            "reasons_to_review": reasons,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.")
