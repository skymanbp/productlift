# AGENTS.md

Tool-agnostic guidance for AI coding agents working in ProductLift. (Claude Code
reads `CLAUDE.md`; this mirrors it for other agents. Keep them in sync.)

## Project
A teaching repo for an applied-ML job search — identify & improve underperforming
e-commerce products on the real Olist dataset. See `README.md` and `LEARNING_PATH.md`.

## Prime directive
This is a **learning repo with graded exercises**. Milestones M0–M8 are implemented
(50/50 tests green) and the `⛏️ EXERCISE` headers stay in place as reference
material. Any newly added exercise is for the learner to implement: **do
not solve it unless explicitly asked** — coach instead: explain the concept, point
to the failing test, review attempts. Infrastructure marked `FULLY BUILT` /
`MOSTLY BUILT` can be edited normally.

## Conventions (enforced)
- **No data leakage**: fit transforms on train only; temporal splits; no
  future/label-derived features. This is the #1 review priority.
- **Validation**: temporal splits, PR-AUC + calibration under imbalance, always a
  baseline first.
- **Reproducibility**: seed from `config/params.yaml`; no magic numbers in code.
- **Statistics**: report effect sizes + CIs (not just p-values); correct for
  multiple testing; state causal assumptions (unconfoundedness, overlap).
- Details in `.claude/rules/`.

## Workflow
Test-first. `make check` (ruff + mypy + pytest) is the gate; `make test-exercise`
runs only the exercise-marked tests. Build the pipeline with
`make features → train → evaluate`; serve with `make serve`.

## Layout
`app/{data,features,models,serving}`, `evaluation/`, `experimentation/`, `causal/`,
`observability/`, `scripts/`, `tests/`, `docs/`. Full map in `README.md`.
