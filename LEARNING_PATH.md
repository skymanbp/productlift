# Learning path

A milestone-by-milestone curriculum. Each milestone = a set of exercise files to
implement, a test that defines "done," and the interview skill it builds.
Milestones 0–8 below are all complete — `NOTES.md` records a commit for M1–M8 (M0
shipped no code), and `git show <sha>` is the answer key. Run `make check` any time
to confirm the tree is still green.

> **How an exercise works:** open the module — it has a `TEST` section, and usually
> `LEARNING GOAL` / `KEY CONCEPT` / `INTERVIEW ANGLE` too (a few omit one or two).
> Read the test (it's the spec). Implement the
> function bodies. Make the test green. Then say it out loud (interview
> prep). The `/fix-issue` command runs this loop with you.

---

## Milestone 0 — Setup & EDA
- **Do:** `make install`, `make data` (download Olist), then work
  `notebooks/01_eda_starter.ipynb`.
- **Skill:** know your data and target before modeling.
- **Done when:** you can state the target's class balance and your top-3 candidate signals.

## Milestone 1 — Data foundations (leakage-safe)
- **Implement:** `app/data/splits.py`, `app/data/labels.py`
- **Tests:** `tests/test_splits.py`, `tests/test_labels.py`
- **Skill:** temporal splits, within-category labels, the leakage mindset.
- **Interview:** "How would you validate this?" / "How did you define the target?"

## Milestone 2 — Feature engineering  ⭐ (her focus)
- **Implement:** `app/features/product.py`, `behavioral.py`, `temporal.py`,
  `encoders.py`, then wire `app/features/build.py`.
- **Tests:** `tests/test_features.py`, `test_encoders.py`, `test_build.py`
- **Skill:** behavioral/temporal/content features, smoothed target encoding without
  leakage, cold-start handling.
- **Run:** `make features` should now produce `data/processed/modeling_matrix.parquet`.

## Milestone 3 — Modeling & validation  ⭐
- **Implement:** `app/models/baseline.py` → `gbdt.py` → `calibrate.py` → (stretch) `tune.py`
- **Tests:** `tests/test_models.py`
- **Skill:** baseline-first discipline, GBDTs, class imbalance, calibration, Optuna.
- **Run:** `make train` registers a model with a ModelCard.

## Milestone 4 — Evaluation
- **Implement:** `evaluation/metrics.py` (Brier, ECE, recall@k, NDCG), `offline_eval.py`
- **Tests:** `tests/test_metrics.py`, `test_offline_eval.py`
- **Skill:** the right metrics under imbalance, calibration reporting, by-segment analysis.
- **Run:** `make evaluate` prints a report and logs to `evaluation/eval_history/`.

## Milestone 5 — Experimentation & statistics  ⭐
- **Implement:** `experimentation/power.py`, `analyze.py`, `segments.py`
- **Tests:** `tests/test_power.py`, `test_ab_analyze.py`, `test_segments.py`
- **Skill:** power analysis, two-proportion tests + CIs, guardrails, multiple-testing.
- **Interview:** the entire back half of the framework lives here.

## Milestone 6 — Causal inference  ⭐
- **Implement:** `causal/propensity.py`, `uplift.py`, `exposure_bias.py`
- **Tests:** `tests/test_propensity.py`, `test_uplift.py`, `test_exposure_bias.py`
- **Skill:** confounding & IPW, per-unit uplift, position/exposure bias — "is the
  product bad or just badly ranked?"

## Milestone 7 — Observability
- **Implement:** `observability/drift.py`
- **Tests:** `tests/test_drift.py`
- **Skill:** PSI drift, retraining triggers.

## Milestone 8 — Production polish
- **Do:** `make serve` then hit `/predict`; `docker compose up --build`; make
  `make check` (lint + types + tests) pass clean.
- **Skill:** shipping a model behind an API, in a container — the "production-ready"
  signal employers look for.

---

### Suggested cadence (with your mentor)
Two milestones per session works well: implement solo, then walk through it out loud
and run `/review` together. Milestones 2, 3, 5, 6 are the deep ones — don't rush them.
See [docs/interview-prep.md](docs/interview-prep.md) for what to say at each step.
