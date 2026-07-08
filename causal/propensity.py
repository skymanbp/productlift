"""Propensity scores & inverse-probability weighting (IPW).  ⛏️  EXERCISE — Milestone 6.

LEARNING GOAL
    Estimate a treatment effect from OBSERVATIONAL data, where treatment wasn't
    randomized (e.g. "does fast delivery cause better reviews?" — but fast delivery
    isn't assigned at random; cheaper/closer orders get it). Propensity weighting
    rebalances the groups to mimic a randomized experiment.

KEY CONCEPT — propensity, IPW, and its assumptions
    Propensity e(x) = P(treated | covariates x), estimated with logistic regression.
    IPW reweights each unit by 1/e(x) (treated) or 1/(1-e(x)) (control) so confounders
    are balanced, then compares weighted outcomes → ATE.
    ASSUMPTIONS you must state: (1) unconfoundedness (you measured the confounders),
    (2) positivity/overlap (every unit could plausibly get either treatment —
    0 < e(x) < 1). Check overlap; CLIP extreme weights or you get huge-variance
    nonsense from a single near-0 propensity.

INTERVIEW ANGLE
    Name the estimand (ATE vs ATT), state both assumptions unprompted, and mention
    weight clipping / trimming. That trio signals real causal literacy.

TEST
    tests/test_propensity.py  (synthetic data with a KNOWN confounded effect)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity(X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
    """Return P(treated | X) for each row via logistic regression.

    Deliberately unscaled and unregularized-beyond-default: the propensity model
    only needs to rank treatment probability given the measured confounders, and
    keeping it a plain LogisticRegression makes the two causal assumptions the
    ONLY moving parts. In production you would scale X and validate the fit
    (calibration of e(x) matters more than its AUC).
    """
    t = np.asarray(treatment)
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("treatment must contain both classes, coded 0/1")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, t)
    return np.asarray(model.predict_proba(X)[:, 1])


def check_overlap(propensity: np.ndarray, treatment: pd.Series, eps: float = 0.01) -> dict:
    """Summarize positivity/overlap: min/max propensity within each arm.

    has_overlap is False as soon as EITHER arm has propensities pinned within
    `eps` of 0 or 1 — those units effectively could never receive the other
    treatment, so no amount of reweighting can say anything about them. This is
    the diagnostic to SHOW before trusting any IPW estimate; running ipw_ate
    without it is how silent extrapolation happens.
    """
    e = np.asarray(propensity, dtype=float)
    t = np.asarray(treatment).astype(bool)
    treated, control = e[t], e[~t]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("both arms must be non-empty to assess overlap")
    report = {
        "treated_min": float(treated.min()),
        "treated_max": float(treated.max()),
        "control_min": float(control.min()),
        "control_max": float(control.max()),
    }
    report["has_overlap"] = bool(
        min(report["treated_min"], report["control_min"]) > eps
        and max(report["treated_max"], report["control_max"]) < 1.0 - eps
    )
    return report


def ipw_ate(
    outcome: pd.Series, treatment: pd.Series, propensity: np.ndarray, clip: float = 0.01
) -> float:
    """Estimate the Average Treatment Effect via inverse-probability weighting.

    Propensities are clipped to [clip, 1-clip] BEFORE inverting: one unit with
    e(x)=0.001 would otherwise carry weight 1000 and dominate the estimate with
    its noise. We use the Hájek (self-normalized) form — weighted means rather
    than the raw Horvitz-Thompson sums — which trades a little bias for much
    lower variance and is the standard practical choice. Valid only under
    unconfoundedness (all confounders measured, here by construction) and
    positivity (check_overlap first).
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment).astype(bool)
    e = np.clip(np.asarray(propensity, dtype=float), clip, 1.0 - clip)

    w_treated = 1.0 / e[t]
    w_control = 1.0 / (1.0 - e[~t])
    mean_treated = float(np.sum(y[t] * w_treated) / np.sum(w_treated))
    mean_control = float(np.sum(y[~t] * w_control) / np.sum(w_control))
    return mean_treated - mean_control
