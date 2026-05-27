# Interview prep — how this repo maps to the applied-science loop

This project is built around the exact framework from the Wayfair mock-interview
session. Each module exists so you can say *"I've actually done this"* — and have the
code to prove it. Use this page to connect what you build to what you'll say.

## The structure to memorize (it IS the repo)

| # | Interview step | Where you do it | What to say |
|---|---|---|---|
| 1 | Clarify business objective | `docs/problem-framing.md` | "Different goals → different ML systems; let me define underperformance and the metric it serves." |
| 2 | Define the ML problem | `app/data/labels.py` | "I framed it as within-category binary classification, but it's also a ranking and a causal question." |
| 3 | Data + EDA | `notebooks/01_eda_starter.ipynb` | "First I'd look at target balance, signal, missingness, and cold-start." |
| 4 | Feature engineering | `app/features/*` | "Behavioral, temporal, content, and category-relative features — fit on train only to avoid leakage." |
| 5 | Baseline | `app/models/baseline.py` | "I always start with a regularized baseline to size the problem and catch leakage." |
| 6 | Model + validation | `app/models/gbdt.py`, `app/data/splits.py` | "GBDTs on tabular data; temporal split; PR-AUC and calibration, not accuracy, under imbalance." |
| 7 | Offline evaluation | `evaluation/*` | "Headline lift over baseline, calibration, and a by-segment breakdown." |
| 8 | Online experimentation | `experimentation/*` | "Offline metrics are proxies; I'd validate with a powered A/B test and guardrails." |
| 9 | Causal / confounding | `causal/*` | "Is it the product or its ranking? Propensity/IPW and uplift separate the two." |
| 10 | Deploy + monitor | `app/main.py`, `observability/*` | "Batch train, real-time serve, monitor drift with PSI and a retraining trigger." |

## High-impact phrases (from the session) and the code that earns them

- *"I'd start with a simple baseline before increasing complexity."* → `models/baseline.py`
- *"Offline metrics are only proxies for business impact."* → `experimentation/` exists for this reason.
- *"I'd validate improvements through controlled experimentation."* → `experimentation/power.py`, `analyze.py`
- *"I'd carefully check for leakage and exposure bias."* → `.claude/skills/leakage-audit/`, `causal/exposure_bias.py`
- *"The framing depends heavily on the business objective."* → `docs/problem-framing.md`
- *"I'd optimize for measurable business outcomes, not just model performance."* → guardrails in `experimentation/analyze.py`
- *"I'd want to understand whether this is fundamentally a ranking problem."* → `ndcg_at_k`, `causal/uplift.py`

## Push questions you should be able to answer (and where you practiced)

- *"What if new products have almost no data?"* → cold start: content features (`features/product.py`), Bayesian smoothing (`features/encoders.py`).
- *"How would you handle class imbalance?"* → `gbdt.py` (scale_pos_weight) + PR-AUC + calibration.
- *"How long should the experiment run?"* → `power.py::days_to_run` (and cover a weekly cycle).
- *"What if conversion improves but returns increase?"* → guardrail metrics (`analyze.py::check_guardrails`).
- *"How do you know it's the product and not its ranking?"* → `causal/exposure_bias.py`, `propensity.py`.
- *"How would you detect drift / when to retrain?"* → `observability/drift.py` (PSI thresholds).

## How to practice
After each milestone, do a 5-minute out-loud "walk me through it" using the row in
the table above. The `/review` and `/new-experiment` commands end by asking you an
interview-style question on purpose — answer it before moving on.
