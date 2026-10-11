from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA = Path("data")
OUT = Path("evidence")
OUT.mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
partners = pd.read_csv(DATA / "partners.csv")
products = pd.read_csv(DATA / "products.csv")

# Only decided labels can be used for supervised training.
df = train.dropna(subset=["is_fraud"]).copy()
df["is_fraud"] = df["is_fraud"].astype(int)

# Parse dates and join reference tables.
def make_features(frame):
    x = frame.copy()
    x["submitted_at"] = pd.to_datetime(
        x["submitted_at"], errors="coerce"
    )

    x = x.merge(partners, on="partner_id", how="left")
    x = x.merge(products, on="sku", how="left")

    x["submitted_month"] = x["submitted_at"].dt.month.astype("Int64").astype(str)
    x["submitted_year"] = x["submitted_at"].dt.year.astype("Int64").astype(str)
    x["submitted_weekday"] = x["submitted_at"].dt.dayofweek.astype("Int64").astype(str)

    # Policy change: claims under Rs 2,000 auto-approved from May 2026.
    x["post_policy_change"] = (
        x["submitted_at"] >= pd.Timestamp("2026-05-01")
    ).astype(int)

    x["partner_tenure_days"] = (
        x["submitted_at"] -
        pd.to_datetime(x["onboarded_date"], errors="coerce")
    ).dt.days

    # Serial formatting can vary. Keep only a normalized version
    # for quality indicators, not the raw high-cardinality serial.
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

    # Do not use claim_id, raw serial, or inspector_note as predictors.
    return x

df = make_features(df)

# Temporal validation: train before April; validate April-June 2026.
cutoff = pd.Timestamp("2026-04-01")
valid_end = pd.Timestamp("2026-07-01")

validation = df[
    (df["submitted_at"] >= cutoff) &
    (df["submitted_at"] < valid_end)
].copy()

training = df[df["submitted_at"] < cutoff].copy()

# Prevent a repeated claim ID from crossing the split.
validation_ids = set(validation["claim_id"])
training = training[
    ~training["claim_id"].isin(validation_ids)
].copy()

numeric_features = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "list_price_inr",
    "warranty_months",
    "partner_tenure_days",
    "serial_length",
    "serial_has_letters",
    "serial_has_digits",
    "description_length",
    "description_digit_count",
    "post_policy_change",
]

categorical_features = [
    "partner_id",
    "sku",
    "city",
    "partner_type",
    "family",
    "photo_attached",
    "partner_inspected",
    "source",
    "submitted_month",
    "submitted_year",
    "submitted_weekday",
]

text_feature = "description_text"

def make_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
        ("description", TfidfVectorizer(
            max_features=1500, ngram_range=(1, 2)
        ), text_feature),
    ])

    return Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

model = make_pipeline()

X_train = training
y_train = training["is_fraud"]
X_valid = validation
y_valid = validation["is_fraud"]

print("Training rows:", len(training))
print("Validation rows:", len(validation))
print("Training fraud rate:", round(y_train.mean(), 4))
print("Validation fraud rate:", round(y_valid.mean(), 4))

if y_train.nunique() < 2 or y_valid.nunique() < 2:
    raise ValueError("Training and validation must each contain both classes.")

model.fit(X_train, y_train)
prob = model.predict_proba(X_valid)[:, 1]

# Evaluate ranking separately from the eventual review threshold.
pred = (prob >= 0.5).astype(int)

metrics = {
    "average_precision": float(average_precision_score(y_valid, prob)),
    "precision_at_0_5": float(precision_score(y_valid, pred, zero_division=0)),
    "recall_at_0_5": float(recall_score(y_valid, pred, zero_division=0)),
    "f1_at_0_5": float(f1_score(y_valid, pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_valid, pred).tolist(),
    "train_rows": int(len(training)),
    "validation_rows": int(len(validation)),
    "train_fraud_rate": float(y_train.mean()),
    "validation_fraud_rate": float(y_valid.mean()),
}

# Monthly capacity simulation: at most 40 claims per month.
review = validation[
    ["claim_id", "submitted_at", "claim_amount_inr", "is_fraud"]
].copy()

review["risk_score"] = prob
review["review_month"] = (
    pd.to_datetime(review["submitted_at"]).dt.to_period("M").astype(str)
)

monthly_results = []

for month, group in review.groupby("review_month"):
    top = group.nlargest(min(40, len(group)), "risk_score")
    fraud_rows = top[top["is_fraud"] == 1]

    monthly_results.append({
        "month": month,
        "claims_available": len(group),
        "claims_reviewed": len(top),
        "frauds_found": int(len(fraud_rows)),
        "fraud_recall_in_month": (
            float(len(fraud_rows) / group["is_fraud"].sum())
            if group["is_fraud"].sum() else 0.0
        ),
        "fraud_amount_identified_inr": float(
            fraud_rows["claim_amount_inr"].sum()
        ),
        "precision_at_40": float(top["is_fraud"].mean()),
        "random_baseline_expected_frauds": float(
            min(40, len(group)) * group["is_fraud"].mean()
        ),
    })

monthly_df = pd.DataFrame(monthly_results)
monthly_df.to_csv(
    "evidence/monthly_top40_simulation.csv", index=False
)

print("\n--- MONTHLY TOP-40 SIMULATION ---")
print(monthly_df.to_string(index=False))


with open(OUT / "rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

review.to_csv(OUT / "validation_ranked_claims.csv", index=False)
joblib.dump(model, "models/fraud_model_rf.joblib")

print("\n--- BASELINE METRICS ---")
print(json.dumps(metrics, indent=2))
print("\n--- CLASSIFICATION REPORT AT 0.5 ---")
print(classification_report(y_valid, pred, zero_division=0))
print("\nSaved model: models/fraud_model_rf.joblib")
print("Saved metrics: evidence/baseline_metrics.json")


# Generate predictions for the unlabelled test claims.
test_df = pd.read_csv(DATA / "test_unlabelled.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

test_features = make_features(test_df)
test_scores = model.predict_proba(test_features)[:, 1]

scored = pd.DataFrame({
    "claim_id": test_df["claim_id"],
    "score": test_scores,
})

# Preserve the exact sample-submission order and schema.
predictions = sample[["claim_id"]].merge(
    scored, on="claim_id", how="left", validate="one_to_one"
)

if predictions["score"].isna().any():
    raise ValueError("Some test claims are missing prediction scores.")

if len(predictions) != len(test_df):
    raise ValueError("Prediction row count does not match test data.")

if not predictions["score"].between(0, 1).all():
    raise ValueError("Risk scores must be between 0 and 1.")

# Assignment explicitly requires this file in the repository root.
predictions.to_csv("predictions.csv", index=False)

print("\n--- PREDICTIONS CHECK ---")
print("Rows:", len(predictions))
print("Columns:", predictions.columns.tolist())
print("Missing scores:", predictions["score"].isna().sum())
print("Score range:",
      predictions["score"].min(),
      predictions["score"].max())
print(predictions.head())
print("Saved predictions.csv in repository root.")
