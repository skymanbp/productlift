"""Tests for causal/uplift.py (Milestone 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from causal.uplift import TLearner

pytestmark = pytest.mark.exercise


def test_tlearner_detects_heterogeneous_effect():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.binomial(1, 0.5, n)                 # subgroup
    t = rng.binomial(1, 0.5, n)                 # randomized treatment
    cate = 0.05 + 0.40 * x                      # effect: 0.05 if x=0, 0.45 if x=1
    p = np.clip(0.2 + cate * t, 0, 1)
    y = rng.binomial(1, p)

    X = pd.DataFrame({"x": x})
    tl = TLearner(lambda: LogisticRegression()).fit(X, pd.Series(t), pd.Series(y))
    uplift = tl.predict_uplift(X)

    # the high-effect subgroup (x=1) should get a higher estimated uplift
    assert uplift[x == 1].mean() > uplift[x == 0].mean()
