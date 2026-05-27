"""Tests for experimentation/analyze.py (Milestone 5)."""

from __future__ import annotations

import pytest

from experimentation.analyze import two_proportion_test

pytestmark = pytest.mark.exercise


def test_matches_statsmodels_pvalue():
    # control 300/5000 = 6.0% ; treatment 360/5000 = 7.2%
    res = two_proportion_test(300, 5000, 360, 5000, alpha=0.05)

    from statsmodels.stats.proportion import proportions_ztest

    _, p_ref = proportions_ztest([360, 300], [5000, 5000])  # pooled, two-sided
    assert res.p_value == pytest.approx(p_ref, rel=0.05)
    assert res.relative_lift == pytest.approx(0.2, abs=0.02)  # 7.2 vs 6.0 = +20%
    assert res.ci_low <= res.absolute_diff <= res.ci_high


def test_no_difference_is_not_significant():
    res = two_proportion_test(300, 5000, 300, 5000, alpha=0.05)
    assert res.significant is False
    assert res.absolute_diff == pytest.approx(0.0, abs=1e-9)
