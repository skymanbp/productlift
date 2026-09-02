# Architecture

ProductLift mirrors a real applied-ML system: an **offline training pipeline** that
produces a model artifact, and an **online serving** path that uses it. The two are
deliberately separate (you train in batch, you serve in real time) — a core
production distinction.

## Offline pipeline (the ML lifecycle)
```
 data/raw (Olist CSVs)
      │  app/data/load.py            load + join → enriched order-item table
      ▼
 data/interim/base_table.parquet
      │  app/features/build.py       split at as_of into feature/outcome windows
      ├── feature window ──► app/features/*  ──┐  (product, behavioral, temporal,
      │                                        │   leakage-safe encoders)
      └── outcome window ──► app/data/labels   │  (is_underperforming)
                                  └────────────┴──► join on product_id
      ▼
 data/processed/modeling_matrix.parquet
      │  scripts/train.py            baseline → GBDT → calibrate
      ▼                              app/models/* + registry (model + ModelCard)
 models/current.joblib (+ .card.json)
      │  scripts/evaluate.py         evaluation/* metrics, overall + by-segment
      ▼
 evaluation/eval_history/*.json      tracked over time

 Batch runs off the same artifacts:
      scripts/score.py               score every product → data/processed/
                                     product_scores.csv + intervention_list.csv
      scripts/monitor.py             observability/* PSI, training window vs latest
                                     snapshot; exit 1 on breach = retraining trigger
```

## Online path (serving)
```
 POST /predict ──► app/main.py ──► app/serving/predictor.py ──► models/current.joblib
                                         │
                                  same feature contract as training (ModelCard.feature_names)
                                         ▼
                                  calibrated probability + flag
```
Containerized via `app/Dockerfile`; `docker-compose.yml` runs it. `scripts/healthcheck.py`
backs the container HEALTHCHECK.

## Analysis paths (not in the request path)
- **`experimentation/`** — design (power) and analyze (significance, guardrails,
  segments) the A/B test that validates a model-driven intervention in production.
- **`causal/`** — estimate *causal* effects from observational data (propensity/IPW),
  per-unit uplift (who to treat), and exposure/position-bias correction.
- **`observability/`** — PSI drift monitoring on live data → retraining trigger
  (`scripts/monitor.py`, exit 1 on breach).

## Why this shape
Each stage is a pure-ish function with a typed boundary, so it's testable in
isolation (`tests/` mirrors the tree) and composable in `scripts/`. The
training/serving symmetry — same feature names, preprocessing inside the sklearn
Pipeline — is what prevents train/serve skew, a top cause of silent model failure.
