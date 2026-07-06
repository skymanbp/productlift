"""Probability calibration + reliability diagnostics.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    A model that ranks well can still lie about probabilities. If the business acts
    on "P(underperform) = 0.7" — prioritizing review queues, sizing interventions —
    the 0.7 must MEAN 70%. Calibration is what makes scores decision-grade.

KEY CONCEPT — calibrate on data the model never touched
    Fitting the calibrator on the training fold recalibrates the model's overfit
    self-confidence, not its honest error. We use a dedicated calib fold that the
    GBDT neither trained on nor early-stopped on. Platt scaling (sigmoid), not
    isotonic: with ~10^2 calibration rows isotonic's step function overfits; the
    sigmoid's two parameters are learnable from little data.

INTERVIEW ANGLE
    "Your model says 0.7 — do you trust it?" → show the reliability table: bin the
    predictions, compare mean predicted vs observed frequency per bin. Report
    Brier/ECE next to PR-AUC, always.

TEST
    tests/test_models.py::test_reliability_table_on_perfect_calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(
    model: BaseEstimator, X_calib: pd.DataFrame | np.ndarray, y_calib: pd.Series
) -> CalibratedClassifierCV:
    """Wrap a FITTED model with Platt scaling learned on the calib fold.

    FrozenEstimator is sklearn 1.9's replacement for the removed cv="prefit": it
    freezes the model so CalibratedClassifierCV fits only the sigmoid on top,
    never refitting the model on calibration data.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins × [mean_predicted, observed_rate, count].

    Column order is part of the contract (tests and reports read positionally:
    predicted first, observed second). Empty bins keep NaN rather than 0 — "no
    predictions landed here" and "observed rate 0" are different facts.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against interior edges → bin ids 0..n_bins-1, with 1.0 in the last bin
    bin_ids = np.digitize(y_prob_arr, edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            rows.append((y_prob_arr[mask].mean(), y_true_arr[mask].mean(), int(mask.sum())))
        else:
            rows.append((np.nan, np.nan, 0))
    return pd.DataFrame(rows, columns=["mean_predicted", "observed_rate", "count"])
