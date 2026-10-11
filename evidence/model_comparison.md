# Model Comparison

## Validation setup

- Training rows: 9,539
- Validation rows: 2,270
- Validation period: April–June 2026
- Confirmed fraud cases in validation: 46

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 93.0% | 97.8% |
| Average precision | 4.34% | 4.76% |
| Fraud precision at 0.5 | 2.4% | 28.6% |
| Fraud recall at 0.5 | 6.5% | 4.3% |

## Interpretation

Random Forest achieved higher accuracy and average precision, but detected fewer known fraud cases at the default 0.5 threshold. Neither model has demonstrated sufficiently reliable fraud detection for automatic claim decisions.

## Limitations

- Fraud labels are rare.
- Historical records contain repeated claim IDs.
- Legacy system labels require quality checks.
- The May 2026 policy change affects inspection requirements.
- Monthly review performance is inconsistent.
- Validation results do not establish realised financial savings.

## Next step

Verify the monthly ranking simulation, review false positives and false negatives, and evaluate any revised model on the same temporal holdout before making a deployment decision.