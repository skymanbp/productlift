"""Tests for experimentation/segments.py (Milestone 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experimentation.segments import segment_effects

pytestmark = pytest.mark.exercise


def _make_df(n_segments: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for s in range(n_segments):
        n = 1000
        group = rng.choice(["control", "treatment"], size=n)
        # small uniform effect everywhere
        base = 0.1 + 0.02 * (group == "treatment")
        outcome = (rng.uniform(size=n) < base).astype(int)
        frames.append(pd.DataFrame({"segment": f"s{s}", "group": group, "outcome": outcome}))
    return pd.concat(frames, ignore_index=True)


def test_returns_adjusted_pvalues():
    res = segment_effects(_make_df(4), "segment", "group", "outcome")
    assert {"segment", "p_value", "p_adjusted"} <= set(res.columns)
    # multiple-testing correction never makes a p-value smaller
    assert (res["p_adjusted"] >= res["p_value"] - 1e-9).all()


def test_more_segments_is_stricter():
    # the adjusted p for a comparable raw p should grow as we test more segments
    few = segment_effects(_make_df(2), "segment", "group", "outcome")["p_adjusted"].min()
    many = segment_effects(_make_df(10), "segment", "group", "outcome")["p_adjusted"].min()
    assert many >= few - 1e-9
