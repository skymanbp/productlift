"""Data / prediction drift detection.  ⛏️  EXERCISE — Milestone 7 (lighter).

LEARNING GOAL
    Detect when live data has shifted from the training distribution, the reason
    a model that A/B-tested well can decay months later.

KEY CONCEPT — PSI and KS
    PSI (Population Stability Index): bin a feature, compare train vs live bin
    proportions, PSI = Σ (live% - train%) * ln(live% / train%). Rules of thumb:
    <0.1 stable, 0.1–0.25 moderate, >0.25 significant. KS (scipy.stats.ks_2samp)
    is a distribution-level complement.

INTERVIEW ANGLE
    Track input drift (PSI per feature), prediction drift (score distribution),
    and, once labels arrive, performance drift. Tie a PSI threshold to a
    retraining trigger.

TEST
    tests/test_drift.py
"""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI between a reference (train) and current (live) sample of one feature.

    Bin edges are the expected sample's quantiles, so each bin holds ~1/n_bins of
    the reference and drift shows up as the live sample concentrating in a few
    bins, not as an artifact of equal-width bins on a skewed feature. Outer edges
    are ±inf so live values outside the training range are counted, not clipped.
    Every term (a% - e%) * ln(a%/e%) is >= 0, so PSI is 0 only when the binned
    distributions match.
    """
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if len(expected_arr) == 0 or len(actual_arr) == 0:
        raise ValueError("expected and actual must both be non-empty")
    # A NaN would not raise here, it would give a wrong answer: np.quantile
    # returns NaN edges, np.histogram then drops all mass into the first bin of
    # both samples, and PSI reads ~0 whatever the drift. Refuse it instead.
    if np.isnan(expected_arr).any() or np.isnan(actual_arr).any():
        raise ValueError(
            "PSI is undefined over NaN; drop missing values and track the missing "
            "rate separately (a missingness shift is itself drift)"
        )

    inner_edges = np.quantile(expected_arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

    expected_counts, _ = np.histogram(expected_arr, bins=edges)
    actual_counts, _ = np.histogram(actual_arr, bins=edges)
    expected_pct = expected_counts / len(expected_arr)
    actual_pct = actual_counts / len(actual_arr)

    # Floor empty bins so an empty live bin gives a large finite term, not ln(0).
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
