"""Tests for observability/drift.py (Milestone 7)."""

from __future__ import annotations

import numpy as np
import pytest

from observability.drift import population_stability_index

pytestmark = pytest.mark.exercise


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert population_stability_index(a, b) < 0.1   # stable


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    shifted = rng.normal(2, 1, 5000)
    assert population_stability_index(a, shifted) > 0.25  # significant drift
