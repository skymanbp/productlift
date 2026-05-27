# ML conventions — ProductLift

The rules that keep an applied-ML project honest. These are also the things that
separate "kaggle notebook" code from "I'd trust this in production" code — exactly
the gap this project is teaching.

## The cardinal sin: leakage
- **No information from the future or from the label may touch a feature.**
- Fit every transform (scalers, target encoders, imputers) on TRAIN ONLY, then
  apply to valid/test. Never `fit` on the full dataset. Use sklearn `Pipeline`
  so this is structural, not a thing you remember to do.
- Time series get **temporal** splits, never random. See app/data/splits.py.
- Target encoding is the classic leakage trap — use out-of-fold encoding.

## Reproducibility
- One seed, from config (`params.yaml: seed`), threaded into every estimator,
  split, and sampler. An experiment you can't reproduce is not a result.
- Parameters live in `config/params.yaml`, never as magic numbers in functions.

## Metrics
- Class imbalance is the default here — prefer **PR-AUC** over ROC-AUC, and always
  report **calibration** (Brier / ECE), not just ranking metrics.
- Offline metrics are proxies. Every modeling claim should name the **business
  metric** it's a proxy for and how it'd be validated online (A/B test).

## Modeling discipline
- Always establish a **baseline** (logistic/linear or a simple heuristic) before a
  GBDT. "We beat the baseline by X" is the only meaningful framing.
- Justify complexity by measured lift, not novelty. A LightGBM that ties the
  baseline is a loss, not a win.

## Pandas
- No chained-assignment / `SettingWithCopyWarning`. Return new frames; don't mutate
  inputs in place.
- Keep raw data immutable: `data/raw` is read-only; derived tables go to
  `data/interim` and `data/processed`.
