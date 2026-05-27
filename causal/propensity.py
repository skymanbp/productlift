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


def estimate_propensity(X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
    """Return P(treated | X) for each row via logistic regression.

    TODO(lillian): fit sklearn LogisticRegression(treatment ~ X), return
    predict_proba(X)[:, 1]. (In practice you'd scale X; keep it simple and document.)
    """
    raise NotImplementedError("Implement estimate_propensity — see tests/test_propensity.py")


def check_overlap(propensity: np.ndarray, treatment: pd.Series) -> dict:
    """Summarize positivity/overlap: min/max propensity within each arm.

    TODO(lillian): return {'treated_min','treated_max','control_min','control_max',
    'has_overlap': bool}. has_overlap is False if either arm has propensities pinned
    near 0 or 1 (define a small epsilon). This is the diagnostic you'd SHOW before
    trusting any IPW estimate.
    """
    raise NotImplementedError


def ipw_ate(
    outcome: pd.Series, treatment: pd.Series, propensity: np.ndarray, clip: float = 0.01
) -> float:
    """Estimate the Average Treatment Effect via inverse-probability weighting.

    TODO(lillian):
      - clip propensity to [clip, 1 - clip]  (variance control / positivity guard)
      - weights: 1/e for treated, 1/(1-e) for control
      - ATE = weighted mean outcome (treated) - weighted mean outcome (control)
    On the synthetic test, the naive difference-in-means is biased by the confounder
    and IPW recovers the true effect within tolerance — that contrast is the lesson.
    """
    raise NotImplementedError
