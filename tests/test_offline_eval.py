"""Tests for evaluation/offline_eval.py (Milestone 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.offline_eval import evaluate_predictions

pytestmark = pytest.mark.exercise


def test_overall_and_by_segment():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=600)
    # a score correlated with y so metrics are well-defined
    prob = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, size=600), 0, 1)
    seg = pd.Series(rng.choice(["electronics", "toys"], size=600))

    report = evaluate_predictions(y, prob, segments=seg)
    assert "overall" in report
    assert {"roc_auc", "pr_auc", "brier", "ece"} <= set(report["overall"])
    assert "by_segment" in report
    assert set(report["by_segment"]) == {"electronics", "toys"}
    assert report["n"] == 600
