"""Offline evaluation harness.  ⛏️  EXERCISE — Milestone 4.

LEARNING GOAL
    Build the loop that scores a model on the holdout AND breaks the metrics down by
    segment (category, price band, product tenure). The headline number hides
    failures: a model that's great overall but useless on cold products may not
    serve the business goal. Segmenting is what separates a model report from a
    model number.

THE LOOP
    given y_true, y_prob (+ optional segment series):
      - compute the report_metrics from config on the whole set
      - compute the same metrics within each segment level
      - (optional) compare against a baseline's metrics → lift
      - save a timestamped run to eval_history/ for trend tracking

TEST / RUN
    `make evaluate` (after training) ; tests/test_offline_eval.py (on synthetic preds)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import metrics as M

HISTORY_DIR = Path(__file__).parent / "eval_history"

# name -> callable(y_true, y_score). recall_at_k/ndcg need extra args; the harness
# passes them via functools.partial when you wire this up.
METRIC_FUNCS = {
    "roc_auc": M.roc_auc,
    "pr_auc": M.pr_auc,
    "brier": M.brier_score,
    "ece": M.expected_calibration_error,
}


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, segments: pd.Series | None = None
) -> dict:
    """Compute overall + per-segment metrics and return a structured report dict.

    TODO(lillian):
      - overall: {name: fn(y_true, y_prob) for name, fn in METRIC_FUNCS}, plus
        recall_at_k at the config's recall_k_fraction.
      - if `segments` is provided: for each unique level, compute the same metrics
        on that subset (guard tiny/one-class segments — they break AUC; skip or NaN
        with a note).
      - return {"overall": {...}, "by_segment": {level: {...}}, "n": len(y_true)}.
    The test checks overall keys exist and a 2-segment split is reported separately.
    """
    raise NotImplementedError("Implement evaluate_predictions — see tests/test_offline_eval.py")


def save_run(report: dict) -> Path:
    """FULLY BUILT — append a timestamped eval run for trend tracking."""
    HISTORY_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = HISTORY_DIR / f"run_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return path
