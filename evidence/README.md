# Kestrel Home — Warranty Claim Risk Review

A machine-learning prototype for prioritising warranty claims for human investigation. The system returns a risk score and review notes through a FastAPI backend and a Streamlit interface.

**Status:** Working prototype; not approved for automatic claim rejection.

## 1. Project Overview

Kestrel Home receives warranty claims through its online channels and service partners. This project uses historical investigation outcomes to estimate fraud risk for new claims.

The objectives are to:
- Rank claims by estimated fraud risk.
- Provide a simple interface for reviewing a single claim.
- Evaluate model performance on later historical claims.
- Assess a monthly investigation capacity of 40 claims.
- Document model limitations and potential business impact.

A high score is a signal for review, not proof of fraud.

## 2. Project Structure

```text
kestrel-warranty-review/
├── data/                       # Local input data; do not publish
│   ├── train.csv
│   ├── test_unlabelled.csv
│   ├── sample_submission.csv
│   ├── partners.csv
│   └── products.csv
├── src/
│   ├── train_model.py
│   └── api.py
├── models/
│   └── fraud_model_rf.joblib
├── evidence/
│   ├── repeated_claims_audit.csv
│   ├── rf_metrics.json
│   ├── monthly_top40_simulation.csv
│   └── validation_ranked_claims.csv
├── app.py
├── predictions.csv
├── memo.md
├── requirements.txt
├── .gitignore
└── README.md
```

The evidence filenames above describe the intended final structure. Ensure every listed file exists before submission.

## 3. Requirements

- Python 3.10 or a compatible Python version supported by the installed packages
- pip
- A local terminal
- No paid model API key is required

## 4. Setup

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the supplied CSV files in `data/`. Do not upload the raw claims or other operational data to a public repository.

## 5. Train the Model

From the project root:

```powershell
python src/train_model.py
```

The training script uses historical decided outcomes, creates derived features, evaluates the model on a later time period, and saves the fitted model and evaluation outputs.

The Random Forest candidate is saved as:

```text
models/fraud_model_rf.joblib
```

Check the console output and `evidence/` before using the model. Keep the Logistic Regression baseline metrics separate from Random Forest metrics.

## 6. Start the API

In Terminal 1:

```powershell
uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

- `GET /health` — checks that the service is running.
- `POST /predict` — accepts one claim as JSON and returns a risk score and review notes.

Example request:

```json
{
  "claim_id": "DEMO-001",
  "submitted_at": "2026-07-15 10:30:00",
  "partner_id": "SP3150",
  "sku": "KH-RV-03",
  "product_serial": "KH222151083",
  "days_since_purchase": 60,
  "claim_amount_inr": 2500,
  "photo_attached": "Y",
  "partner_inspected": "Y",
  "claim_description": "Product stopped working.",
  "customer_prior_claims": 0,
  "source": "crm"
}
```

The response includes the claim ID, estimated risk score, interpretation and review notes.

## 7. Start the Streamlit Interface

In Terminal 2:

```powershell
streamlit run app.py
```

Enter the claim details and select **Assess claim**. The interface sends the record to the local API and displays the returned score and review notes.

Start the API before using the interface.

## 8. Predictions File

The submission file is `predictions.csv` in the repository root.

Required columns:

```text
claim_id,score
```

The file must contain one row for every record in `test_unlabelled.csv`, preserve the sample submission's claim IDs and ordering, and contain valid numeric scores between 0 and 1.

Higher scores indicate higher estimated fraud risk. The scores are not confirmed fraud probabilities unless separately calibrated and validated.

## 9. Evaluation

The initial temporal validation used historical decided claims, with training before April 2026 and validation from April through June 2026.

Reported Random Forest results:

- Accuracy: approximately 97.8%
- Average precision: approximately 4.76%
- Fraud precision at a 0.5 threshold: approximately 28.6%
- Fraud recall at a 0.5 threshold: approximately 4.3%

The monthly top-40 simulation identified three known fraud claims in April, none in May, and none in June. The three April claims had a combined claimed amount of ₹17,060.

These results are preliminary. They do not establish realised savings or reliable performance on future claims. Verify that all saved metrics and monthly simulations correspond to the same fitted model before submission.

## 10. Limitations and Responsible Use

- Fraud is rare in the labelled data, so accuracy alone is misleading.
- Historical data includes repeated claim IDs and requires careful validation.
- Legacy labels may have quality limitations.
- The May 2026 policy change affects inspection requirements for claims under ₹2,000.
- Partner age alone is not evidence of fraud.
- The current model should not automatically reject claims or delay payouts without human review.
- The model has not demonstrated consistent fraud detection across the validation months.

## 11. Business Recommendation

Use the current system only as an experimental prioritisation aid after its validation and ranking outputs have been verified. Keep reviews within the investigation team's limit of 40 claims per month.

Track confirmed frauds, false positives, claim amounts, genuine-customer delays, and goodwill costs before deciding whether to continue or expand a pilot.

## 12. AI Tools and Cost Disclosure

This project was developed with assistance from ChatGPT for coding guidance, debugging, evaluation interpretation and documentation. The implementation uses local Python libraries and does not require a paid model API key.

Disclose the actual tools and any applicable subscription or API costs in the required submission form. Do not claim that no tools were used.

## 13. Final Checks

- [ ] Dependencies install in a fresh virtual environment.
- [ ] Model training completes successfully.
- [ ] `/health` and `/predict` work.
- [ ] Streamlit successfully calls the API.
- [ ] API and batch prediction feature transformations match.
- [ ] `predictions.csv` has the exact required columns, row count and claim IDs.
- [ ] Validation metrics and monthly simulation match the final model.
- [ ] `memo.md` and required submission form are complete.
- [ ] No raw claims data, secrets or API keys are committed