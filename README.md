# ProductLift 🛒📈

**An applied-ML project that identifies and improves underperforming e-commerce
products — built as a hands-on curriculum for breaking into Applied Scientist / ML
roles.**

ProductLift takes the real [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and walks the full applied-ML lifecycle the way an interviewer expects you to:
business framing → feature engineering → modeling & validation → evaluation →
experimentation → causal inference → production serving & monitoring.

It is a **teaching repo**: the infrastructure is built, and the ML/stats logic is
left as **graded exercises** — each with a failing test that defines "done." You
make the tests pass; the code you write *is* your portfolio and your interview prep.

> Built for a mentor + mentee to work through together. The running case study is
> the Wayfair-style prompt *"identify and improve underperforming products."* See
> [docs/interview-prep.md](docs/interview-prep.md) for how every module maps to an
> interview talking point.

---

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e ".[dev]"          # or: make install

# 2. Get the data (see data/README.md for the Kaggle token, or download manually)
make data

# 3. See your to-do list: every failing test is an exercise to implement
make test-exercise

# 4. Work a milestone, then run the pipeline end to end
make features     # build the modeling matrix   (after Milestones 1–2)
make train        # baseline → GBDT → calibrate  (after Milestone 3)
make evaluate     # metrics + by-segment report  (after Milestone 4)

# 5. Ship it
make serve        # FastAPI scoring service at http://localhost:8000/docs
docker compose up --build
```

Activate the AI-agent guardrails (optional): copy `.claude/settings.json.example`
to `.claude/settings.json`.

## The curriculum

Follow **[LEARNING_PATH.md](LEARNING_PATH.md)** — 8 milestones, each with files to
implement, tests to pass, and the interview skill it builds. The four deep modules
(starred) are **feature engineering, modeling & validation, experimentation &
stats, and causal inference**.

```
Milestone 0  Setup & EDA
Milestone 1  Data foundations (temporal splits, leakage-safe labels)
Milestone 2  Feature engineering ⭐
Milestone 3  Modeling & validation ⭐  (baseline → LightGBM → calibration)
Milestone 4  Evaluation (PR-AUC, calibration, by-segment)
Milestone 5  Experimentation & statistics ⭐  (power, A/B, guardrails)
Milestone 6  Causal inference ⭐  (propensity/IPW, uplift, exposure bias)
Milestone 7  Observability (drift)
Milestone 8  Production polish (serve + containerize)
```

## Repository layout

```
productlift/
├── app/
│   ├── data/         load Olist, leakage-safe splits, the target definition
│   ├── features/     product / behavioral / temporal features + target encoder
│   ├── models/       baseline, LightGBM, calibration, tuning, registry
│   ├── serving/      load a model and score (FastAPI predictor)
│   ├── config.py     env + params.yaml (single source of truth)
│   └── main.py       FastAPI scoring service
├── evaluation/       metrics (AUC/PR-AUC/Brier/ECE/NDCG) + offline harness
├── experimentation/  power analysis, A/B test analysis, segment effects
├── causal/           propensity/IPW, uplift meta-learners, exposure-bias
├── observability/    PSI/KS drift detection + monitoring
├── scripts/          download_data, build_features, train, evaluate, healthcheck
├── notebooks/        guided EDA
├── tests/            one exercise test per module (the specs)
├── docs/             problem-framing, data-dictionary, architecture, interview-prep
├── config/params.yaml
└── .claude/          AI-agent control center (commands, rules, skills, agents)
```

## What it's teaching (the gaps it closes)

| Gap | How the repo addresses it |
|---|---|
| **Production-ready code** | sklearn Pipelines, model registry + cards, FastAPI serving, Docker, CI-ready tests, drift monitoring |
| **Data Science rigor** | leakage discipline, temporal validation, calibration, proper metrics under imbalance |
| **Experimentation** | power analysis, A/B analysis with CIs + guardrails + multiple-testing |
| **Causal thinking** | propensity/IPW, uplift, exposure/position-bias correction |
| **Communication** | every module has an `INTERVIEW ANGLE`; `docs/interview-prep.md` rehearses the story |

## Tech choices (and why)
- **LightGBM/XGBoost + scikit-learn** — GBDTs are the right default on tabular data.
- **Olist real data** — messy and real beats synthetic for portfolio credibility.
- **statsmodels / scipy** — first-class stats for the experimentation module.
- **FastAPI + Docker** — the standard, lightweight way to ship a model.

## A note on honesty
Olist is order-level (no impression/traffic logs), so we adapt "underperformance"
to what the data supports and study causal questions the data *can* answer. The
reasoning is spelled out in [docs/problem-framing.md](docs/problem-framing.md) — and
being able to explain that adaptation is itself an interview-grade skill.
