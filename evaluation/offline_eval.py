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
from datetime import UTC, datetime
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


def _metric_suite(y: np.ndarray, p: np.ndarray, k_fraction: float) -> dict[str, float | None]:
    """One row of the report. Rank metrics (AUCs, recall@k) need both classes
    present — on a one-class slice they are undefined, reported as None rather
    than a fabricated number. Brier/ECE stay defined either way."""
    single_class = len(np.unique(y)) < 2
    out: dict[str, float | None] = {}
    for name, fn in METRIC_FUNCS.items():
        rank_metric = name in {"roc_auc", "pr_auc"}
        out[name] = None if (single_class and rank_metric) else float(fn(y, p))
    out["recall_at_k"] = None if single_class else float(M.recall_at_k(y, p, k_fraction))
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segments: pd.Series | None = None,
    *,
    min_segment_size: int = 30,
) -> dict:
    """Compute overall + per-segment metrics and return a structured report dict.

    Segment levels with fewer than `min_segment_size` rows are skipped outright:
    an AUC over a dozen rows is noise wearing a number's clothes, and reporting it
    invites decisions based on it. recall@k uses the config's recall_k_fraction so
    the report answers the operating question the business actually set.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    from app.config import (
        get_params,
    )  # local import: keep evaluation importable without app config side effects

    k_fraction = float(get_params()["evaluation"]["recall_k_fraction"])

    report: dict = {"overall": _metric_suite(y, p, k_fraction), "n": int(len(y))}
    if segments is not None:
        seg = pd.Series(segments).reset_index(drop=True)
        by_segment: dict[str, dict[str, float | None]] = {}
        for level in seg.dropna().unique():
            mask = (seg == level).to_numpy()
            if int(mask.sum()) < min_segment_size:
                continue
            by_segment[str(level)] = _metric_suite(y[mask], p[mask], k_fraction)
        report["by_segment"] = by_segment
    return report


def save_run(report: dict) -> Path:
    """FULLY BUILT — append a timestamped eval run for trend tracking."""
    HISTORY_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = HISTORY_DIR / f"run_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return path
