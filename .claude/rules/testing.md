# Testing — ProductLift

Testing ML code is its own skill, and a big part of the "production-ready" gap.

## Philosophy
- **The test is the spec.** Each exercise ships a failing test that defines done.
- Tests run on tiny **fixtures**, never the real Olist dataset — fast and
  deterministic. Tests that need the download are marked `@pytest.mark.data`.
- Exercise tests are marked `@pytest.mark.exercise` → `make test-exercise` is the
  daily to-do list.

## What to test in ML code
- **Leakage**: assert a transform fitted on train doesn't peek at valid/test
  (e.g. a target encoder gives the global prior for an unseen category).
- **Shape & invariants**: row counts after a join, no surprise NaNs, probabilities
  in [0, 1], monotonic where expected.
- **Metric correctness**: hand-compute on a 3–5 row example and assert exact values.
  An off-by-one in a metric silently corrupts every experiment.
- **Determinism**: same seed → same output.
- **Temporal correctness**: a split's train max-date < valid min-date.

## Don't
- Don't assert on exact model scores (they drift across library versions). Assert
  on properties: "calibrated model has lower Brier than uncalibrated", "AUC > 0.5".
- Don't fit on the whole dataset in a test and call it validation.
