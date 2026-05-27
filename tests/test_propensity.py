"""Tests for causal/propensity.py (Milestone 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal.propensity import check_overlap, estimate_propensity, ipw_ate

pytestmark = pytest.mark.exercise


@pytest.fixture
def confounded():
    """A confounder z drives BOTH treatment assignment and the outcome, so the naive
    difference-in-means is biased. True treatment effect = 1.0."""
    rng = np.random.default_rng(0)
    n = 5000
    z = rng.binomial(1, 0.5, n)
    p_treat = 0.2 + 0.6 * z            # treated more often when z=1 (confounding)
    t = rng.binomial(1, p_treat)
    tau = 1.0
    y = 2.0 * z + tau * t + rng.normal(0, 1, n)
    return pd.DataFrame({"z": z, "t": t, "y": y}), tau


def test_propensity_in_unit_interval(confounded):
    df, _ = confounded
    e = estimate_propensity(df[["z"]], df["t"])
    assert np.all((e > 0) & (e < 1))


def test_ipw_recovers_effect_naive_does_not(confounded):
    df, tau = confounded
    naive = df.loc[df.t == 1, "y"].mean() - df.loc[df.t == 0, "y"].mean()
    e = estimate_propensity(df[["z"]], df["t"])
    ate = ipw_ate(df["y"], df["t"], e)
    assert abs(ate - tau) < 0.25          # IPW corrects the confounding
    assert (naive - tau) > 0.25           # naive is biased upward by z


def test_overlap_report(confounded):
    df, _ = confounded
    e = estimate_propensity(df[["z"]], df["t"])
    report = check_overlap(e, df["t"])
    assert "has_overlap" in report
