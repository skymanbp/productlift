"""Data / prediction drift detection.  ⛏️  EXERCISE — Milestone 7 (lighter).

LEARNING GOAL
    Detect when live data has shifted from the training distribution — the doc lists
    "drift detection" and "retraining cadence" as production must-haves. Drift is why
    a model that A/B-tested great slowly decays months later (the holidays example in
    the doc).

KEY CONCEPT — PSI and KS
    - Population Stability Index (PSI): bin a feature, compare train vs live bin
      proportions: PSI = Σ (live% - train%) * ln(live% / train%). Rules of thumb:
      <0.1 stable, 0.1–0.25 moderate shift, >0.25 significant — investigate/retrain.
    - KS statistic: max gap between the two empirical CDFs (scipy.stats.ks_2samp) —
      a distribution-level test complementing PSI.

INTERVIEW ANGLE
    "How would you monitor it?" → track input drift (PSI per feature), prediction
    drift (score distribution), and — when labels eventually arrive — performance
    drift. Tie a PSI threshold to a retraining trigger.

TEST
    tests/test_drift.py
"""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Return PSI between a reference (train) and current (live) sample of one feature.

    TODO(lillian):
      - define bin edges from the EXPECTED distribution's quantiles (so bins are
        balanced on the reference)
      - compute the proportion of each sample falling in each bin
      - PSI = Σ (actual% - expected%) * ln(actual% / expected%)
      - add a tiny epsilon to avoid log(0) / divide-by-zero on empty bins.
    The test checks PSI≈0 for identical samples and a large PSI for a shifted one.
    """
    raise NotImplementedError("Implement population_stability_index — see tests/test_drift.py")
