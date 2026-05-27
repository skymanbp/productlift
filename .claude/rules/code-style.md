# Code style — ProductLift

## Python
- Python 3.11+, type hints on every function. `mypy` must pass.
- Format/lint with **ruff** (`make format`, `make lint`).
- Prefer **pure functions** for transforms and metrics — input frame in, new frame
  out, no hidden state. Pure functions are trivially testable, which is the whole
  game.
- Keep functions small and named after the domain: `recall_at_k`, `target_encode`,
  `temporal_split` — not `process`, `transform_data`, `do_it`.

## Docstrings
- Every module opens with a docstring. Exercise modules use the template:
  `LEARNING GOAL`, `KEY CONCEPT`, `INTERVIEW ANGLE`, `TEST`.
- Document *why* (the statistical/business reason), not *what*.

## Numbers & config
- Magic numbers (quantiles, windows, thresholds, seed) come from
  `config/params.yaml`, never hardcoded.

## Data frames
- Don't mutate inputs in place; return new frames.
- Validate assumptions early (expected columns, no unexpected NaNs) and fail loud.
