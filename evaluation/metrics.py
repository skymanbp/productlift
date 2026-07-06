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

    Lower = better-calibrated AND sharper (it decomposes into calibration +
    refinement); a constant prediction of the prevalence scores prevalence*(1-prevalence).
    """
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: bin predictions, take the weighted average gap between mean predicted
    probability and observed frequency per bin.

    Weighting by bin population keeps a sparsely-hit bin from dominating the
    score; empty bins contribute nothing (there is no prediction to be wrong).
    This is THE metric to show alongside AUC in a model review.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against interior edges → bin ids 0..n_bins-1, with 1.0 in the last bin
    bin_ids = np.digitize(y_prob_arr, edges[1:-1])

    n = len(y_prob_arr)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            gap = abs(y_prob_arr[mask].mean() - y_true_arr[mask].mean())
            ece += (mask.sum() / n) * gap
    return float(ece)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Of all true positives, what share is captured in the top `k_fraction` by score?

    This is the operating-point metric — "if we can only review the top 10% of
    flagged products, how many real underperformers do we catch?" It translates
    the model into a business decision. NaN when there are no positives: recall
    is undefined without anything to recall, and 0/1 would both lie.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    n_pos = int(y_true_arr.sum())
    if n_pos == 0:
        return float("nan")
    k = int(np.ceil(k_fraction * len(y_true_arr)))
    top_k = np.argsort(-y_score_arr, kind="stable")[:k]
    return float(y_true_arr[top_k].sum() / n_pos)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain — a ranking metric (the doc lists MAP/
    NDCG). Binary relevance: gain = y_true.

    DCG@k = Σ_{i=1..k} rel_i / log2(i+1) over the score-descending order; IDCG@k is
    the same sum over the ideal (relevance-descending) order; NDCG = DCG/IDCG.
    0 when IDCG == 0 — with no relevant items every ranking is equally (un)helpful.
    Relevant for the ranking framing of "which products to surface/fix first".
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score_arr, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((y_true_arr[order] * discounts).sum())

    ideal = np.sort(y_true_arr)[::-1][:k]
    idcg = float((ideal * 1.0 / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0
