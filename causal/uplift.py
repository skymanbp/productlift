"""Uplift modeling (heterogeneous treatment effects).  ⛏️  EXERCISE — Milestone 6.

LEARNING GOAL
    Estimate the effect of an intervention PER UNIT, not on average. For "which
    underperforming products should we promote?" the right target isn't "who will
    convert" but "whose conversion will INCREASE BECAUSE of the promotion" — the
    uplift. Targeting by uplift instead of by predicted outcome is a big practical win.

KEY CONCEPT — meta-learners (T-learner, S-learner)
    - S-learner: one model with treatment as a feature; CATE(x) = f(x, 1) - f(x, 0).
    - T-learner: two models, one per arm; CATE(x) = f_treated(x) - f_control(x).
    Both estimate the Conditional Average Treatment Effect (CATE). T-learner is the
    cleaner first implementation.

INTERVIEW ANGLE
    Distinguish "propensity to convert" from "uplift from treatment" — promoting
    sure-things wastes budget; promoting persuadables creates value. The four-quadrant
    framing (sure things / persuadables / lost causes / sleeping dogs) is gold.

TEST
    tests/test_uplift.py  (synthetic data where the true CATE differs by subgroup)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TLearner:
    """Two-model uplift estimator. `model_factory` returns a fresh regressor/
    classifier each call (so the two arms get independent models)."""

    def __init__(self, model_factory) -> None:
        self._make = model_factory
        self.model_t = None
        self.model_c = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "TLearner":
        """Fit one model on treated rows, one on control rows.

        TODO(lillian): split X/outcome by treatment, fit self.model_t on treated and
        self.model_c on control (each via self._make()). Return self.
        """
        raise NotImplementedError("Implement TLearner.fit — see tests/test_uplift.py")

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row CATE estimate = pred_treated(X) - pred_control(X).

        TODO(lillian): use predict_proba[:,1] for classifiers (or predict for
        regressors). The test checks that the estimated uplift is higher for the
        subgroup with the larger true effect.
        """
        raise NotImplementedError
