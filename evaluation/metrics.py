"""Metrics. Discrimination metrics are provided; the calibration, operating-point,
and ranking metrics are  ⛏️  EXERCISES — Milestone 4.

WHY THE SPLIT
    roc_auc / pr_auc / rmse are standard library one-liners — provided so the
    harness runs. The ones you implement (Brier, ECE, recall@k, NDCG@k) are the ones
    interviewers probe and the ones the session doc stresses (calibration!). Know
    them cold; an off-by-one in a metric silently corrupts every experiment.

TEST
    tests/test_metrics.py — hand-computed expected values in the comments.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Provided (discrimination / error)
# --------------------------------------------------------------------------- #
def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve. Rank metric: P(score(pos) > score(neg))."""
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision). Preferred under
    class imbalance, where ROC-AUC can look deceptively good."""
    return float(average_precision_score(y_true, y_score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# --------------------------------------------------------------------------- #
# ⛏️ EXERCISES
# --------------------------------------------------------------------------- #
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions: mean((y_prob - y_true)^2).

    TODO(lillian): one line. Lower = better-calibrated AND sharper. The test checks
    a perfect predictor (0.0) and a hand example.
    """
    raise NotImplementedError


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: bin predictions, take the weighted average gap between mean predicted
    probability and observed frequency per bin.

    TODO(lillian):
      - bin y_prob into n_bins equal-width bins on [0, 1]
      - per non-empty bin: |mean(y_prob in bin) - mean(y_true in bin)|
      - weight each bin's gap by its share of points; sum.
    The test feeds a perfectly-calibrated set (ECE≈0) and a badly-calibrated one.
    This is THE metric to show alongside AUC in a model review.
    """
    raise NotImplementedError


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Of all true positives, what share is captured in the top `k_fraction` by score?

    TODO(lillian): take the top ceil(k_fraction * n) items by score; return
    (# true positives among them) / (total true positives). This is the operating-
    point metric — "if we can only review the top 10% of flagged products, how many
    real underperformers do we catch?" Translate the model into a business decision.
    """
    raise NotImplementedError


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain — a ranking metric (the doc lists MAP/
    NDCG). Binary relevance: gain = y_true.

    TODO(lillian): order by y_score desc; DCG@k = Σ_{i=1..k} rel_i / log2(i+1);
    IDCG@k = DCG of the ideal ordering; return DCG/IDCG (0 if IDCG==0). Relevant for
    the ranking framing of "which products to surface/fix first".
    """
    raise NotImplementedError
