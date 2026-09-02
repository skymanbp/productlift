# CLAUDE.md — ProductLift

Context and instructions for Claude Code (and any AI agent) working in this repo.

## What this project is
A **teaching repo** for an applied-ML / Applied Scientist job search. A mentor and a
mentee (strong stats/simulation PhD, new to *production* ML) work through it
together. The running case study: identify & improve underperforming e-commerce
products on the real Olist dataset. Full curriculum: `LEARNING_PATH.md`.

## The single most important rule
**This is a learning repo. Do NOT implement the exercises for the learner unless
explicitly asked.** Milestones M1–M8 are implemented (49/49 tests green), so the rule
governs any newly added exercise. Exercise files are marked `⛏️ EXERCISE`, and their
docstrings usually carry `LEARNING GOAL` / `KEY CONCEPT` / `INTERVIEW ANGLE` / `TEST`
sections — a few modules omit one or two.
When she's working an exercise, your job is to **coach**: explain the concept, point
at the failing test, ask leading questions, review her attempt — not to hand over the
solution. If she explicitly asks you to implement it, do so *and explain it*, since
the explanation is the point.

The infrastructure (anything marked `FULLY BUILT` / `MOSTLY BUILT`) is fair game to
edit, debug, and extend normally.

## How to work here
- **Test-first.** The test is the spec. `make check` (ruff + mypy + pytest) is the
  gate; use the `/fix-issue` command's loop on a new exercise or a regression.
- **Follow the rules** in `.claude/rules/`: `ml-conventions.md` (leakage, validation,
  metrics), `code-style.md`, `testing.md`. These are non-negotiable and double as
  review criteria.
- **Leakage is the cardinal sin** — fit transforms on train only, temporal splits,
  no future/label-derived features. When results look too good, run the
  `leakage-audit` skill before celebrating.
- **Config over magic numbers** — tunables live in `config/params.yaml`.
- **Connect to the business** — every modeling choice names the business metric it
  serves; offline metrics are proxies validated by A/B tests.

## Commands
- `make install | data | features | train | evaluate | serve` — the pipeline.
- `make test | test-exercise | lint | typecheck | check` — quality gates.
- Slash commands: `/review` (interviewer-style diff review), `/fix-issue <test>`
  (work an exercise to green), `/new-experiment <change>` (run a clean experiment).
- Subagents: `ml-reviewer` (modeling/features), `stats-reviewer` (experimentation/
  causal). Skills: `leakage-audit`, `eval-report`.

## Tooling
Python 3.11+, pandas, scikit-learn, LightGBM, statsmodels, scipy, structlog,
FastAPI. Lint/format with ruff; type-check with mypy. Personal overrides go in
`CLAUDE.local.md` (gitignored).
