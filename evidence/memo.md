# Kestrel Home — Warranty Claim Review

**To:** Ritu Deshpande, Head of D2C Operations  
**Subject:** Warranty fraud model — validation results and recommendation

## Recommendation

Do not use the current model to automatically reject warranty claims or approve payouts based on its score alone. Continue with a limited, human-reviewed pilot only after the validation results and review-ranking logic have been checked.

## What we found

The Random Forest model achieved approximately 97.8% accuracy on a temporal validation set of 2,270 claims. However, average precision was only 4.76%, and the model detected 2 of 46 known fraud cases at the default 0.5 threshold. High accuracy therefore does not mean reliable fraud detection.

In a preliminary monthly review simulation, the top 40 claims identified 3 known fraud cases in April, associated with ₹17,060 in claimed amounts. The same approach identified no known fraud in the top 40 claims in May or June. These results are inconsistent and are not evidence of realised savings.

## Financial interpretation

The operations policy estimates that a fraudulent payout costs the full claim amount and that holding a genuine claim for review incurs an average goodwill cost of ₹380. The validation simulation suggests a potential opportunity to prevent some payouts, but it does not establish net savings. The model's performance must improve and remain consistent before financial benefits can be relied upon.

## What to do next week

1. Audit historical labels, repeated submissions and the effect of the May 2026 policy change.
2. Recheck monthly top-40 results against a reproducible validation set.
3. Run a controlled review pilot within the investigation team's capacity of 40 claims per month.
4. Track confirmed frauds, genuine claims held, claim amounts, review outcomes and goodwill costs.
5. Reassess the model before considering wider use.

**Decision:** The current model is an experimental review-prioritisation tool, not a production-ready fraud decision system. No claim should be rejected solely because of its model score.