"""Tests for experimentation/power.py (Milestone 5)."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from experimentation.power import days_to_run, required_sample_size

pytestmark = pytest.mark.exercise


def test_sample_size_matches_formula():
    p1, mde, alpha, power = 0.10, 0.20, 0.05, 0.80
    p2 = p1 * (1 + mde)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    expected = math.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)
    assert required_sample_size(p1, mde, alpha, power).n_per_arm == expected


def test_bigger_effect_needs_smaller_sample():
    small_effect = required_sample_size(0.10, 0.05).n_per_arm
    big_effect = required_sample_size(0.10, 0.20).n_per_arm
    assert big_effect < small_effect


def test_days_to_run_covers_a_week():
    # even with huge daily traffic, run at least one weekly cycle
    assert days_to_run(10, 10_000) >= 7
    assert days_to_run(50_000, 1_000) == pytest.approx(50, abs=1)
