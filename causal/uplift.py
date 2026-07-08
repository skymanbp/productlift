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

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> TLearner:
        """Fit one model on treated rows, one on control rows.

        The two arms get INDEPENDENT models from the factory — sharing one model
        (the S-learner) shrinks small effects toward zero because treatment
        competes with every other feature for splits/coefficients. An empty arm
        is an experiment-design failure, not something to impute around.
        """
        t = np.asarray(treatment).astype(bool)
        if t.all() or not t.any():
            raise ValueError("both treated and control rows are required to fit a T-learner")
        self.model_t = self._make().fit(X[t], np.asarray(outcome)[t])
        self.model_c = self._make().fit(X[~t], np.asarray(outcome)[~t])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row CATE estimate = pred_treated(X) - pred_control(X).

        Classifiers contribute P(y=1) (predict_proba[:, 1]); regressors their
        prediction — so the uplift is always on the outcome scale, never on
        hard 0/1 labels (differencing labels would quantize the effect away).
        """
        if self.model_t is None or self.model_c is None:
            raise ValueError("TLearner.predict_uplift called before fit")
        return self._positive_prediction(self.model_t, X) - self._positive_prediction(
            self.model_c, X
        )

    @staticmethod
    def _positive_prediction(model, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X)[:, 1])
        return np.asarray(model.predict(X), dtype=float)
