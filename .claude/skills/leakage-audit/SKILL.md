---
name: leakage-audit
description: Hunt for data leakage and train/test contamination across the feature and modeling code. Use before trusting any offline metric, and any time results look "too good."
---

# Data leakage audit

Leakage is the #1 reason offline metrics lie. If AUC looks suspiciously high, assume
leakage until proven otherwise. Walk these checks.

## 1. Target leakage (a feature encodes the label)
- Any feature computed using the outcome, or anything that only exists *after* the
  outcome is known (post-purchase data, final review score used to predict
  underperformance, delivery outcome used to predict the same)?
- Aggregates that include the row's own label (e.g. category mean target computed
  over all rows including this one)?

## 2. Temporal leakage (future → past)
- Is the split temporal? Find any random split on time-series data.
- Do features use data from after the prediction timestamp (rolling windows that
  look forward, "lifetime" aggregates computed over the whole history)?

## 3. Fit-on-everything (train/test contamination)
- Is any transformer (`StandardScaler`, target encoder, imputer, PCA) `.fit()` on
  the full dataset before splitting? It must fit on TRAIN only.
- Target encoding without out-of-fold computation? Classic, subtle, high-impact.

## 4. Duplicate / grouped rows across splits
- Could the same product/customer appear in both train and test, leaking identity?
  Should the split be grouped?

## Method
Grep for `.fit(` and `.fit_transform(` and check what data each touches. Trace each
feature back to its source columns and ask "was this knowable at prediction time?"

## Output
A findings list: severity (CRITICAL/HIGH/MEDIUM), `file:line`, the leak mechanism in
one sentence, and the fix. If you find none, say what you checked so the clean bill
of health is credible.
