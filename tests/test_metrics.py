"""Tests for evaluation/metrics.py (Milestone 4). Hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics

pytestmark = pytest.mark.exercise


def test_brier_score():
    y = np.array([1, 0, 1, 0])
    # perfect probabilistic predictions -> 0
    assert metrics.brier_score(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)
    # all 0.5 -> mean((0.5 - y)^2) = 0.25
    assert metrics.brier_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)


def test_ece_perfect_vs_bad():
    rng = np.random.default_rng(0)
    # perfectly calibrated: prob p, outcome ~ Bernoulli(p), large n -> ECE ~ 0
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert metrics.expected_calibration_error(y, p, n_bins=10) < 0.05
    # badly calibrated: always predict 0.99 but truth is ~50/50 -> large ECE
    y2 = (rng.uniform(size=2000) < 0.5).astype(int)
    p2 = np.full(2000, 0.99)
    assert metrics.expected_calibration_error(y2, p2, n_bins=10) > 0.3


def test_recall_at_k():
    # scores rank items; top 50% = 2 items. true positives total = 2 (ids 0 and 3).
    y = np.array([1, 0, 0, 1])
    score = np.array([0.9, 0.2, 0.1, 0.8])  # top-2 by score = indices 0,3 (both positive)
    assert metrics.recall_at_k(y, score, k_fraction=0.5) == pytest.approx(1.0)


def test_ndcg_at_k():
    # relevant at ranks 1 and 3 -> nDCG = 1.5 / 1.6309 ≈ 0.9197 (see RAG-free derivation)
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.7, 0.5, 0.3])
    assert metrics.ndcg_at_k(y, score, k=4) == pytest.approx(0.9197, abs=1e-3)
