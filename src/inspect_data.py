
from pathlib import Path
import pandas as pd

DATA = Path("data")

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test_unlabelled.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")
partners = pd.read_csv(DATA / "partners.csv")
products = pd.read_csv(DATA / "products.csv")

print("\n--- SHAPES ---")
for name, df in [
    ("Train", train),
    ("Test", test),
    ("Sample", sample),
    ("Partners", partners),
    ("Products", products),
]:
    print(name, df.shape)

print("\n--- TRAIN COLUMNS ---")
print(train.columns.tolist())

print("\n--- LABEL COUNTS ---")
print(train["is_fraud"].value_counts(dropna=False))

print("\n--- LABELS BY SOURCE ---")
print(pd.crosstab(
    train["source"],
    train["is_fraud"],
    dropna=False
))

print("\n--- DUPLICATE CLAIM IDS ---")
print("Train:", train["claim_id"].duplicated().sum())
print("Test:", test["claim_id"].duplicated().sum())

print("\n--- MISSING VALUES ---")
print(train.isna().sum().sort_values(ascending=False))

print("\n--- DATE RANGES ---")
for name, df in [("Train", train), ("Test", test)]:
    dates = pd.to_datetime(df["submitted_at"], errors="coerce")
    print(name, dates.min(), dates.max())

print("\n--- FRAUD RATE BY MONTH ---")
train["month"] = pd.to_datetime(
    train["submitted_at"], errors="coerce"
).dt.to_period("M")
print(train.groupby("month")["is_fraud"].agg(
    ["count", "mean"]
).tail(12))

print("\n--- SAMPLE SUBMISSION FORMAT ---")
print(sample.head())
print("Columns:", sample.columns.tolist())

print("\n--- PARTNER / PRODUCT KEYS ---")
print("Unmatched train partners:",
      (~train["partner_id"].isin(partners["partner_id"])).sum())
print("Unmatched train products:",
      (~train["sku"].isin(products["sku"])).sum())
