"""Tests for causal/exposure_bias.py (Milestone 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal.exposure_bias import position_adjusted_rate

pytestmark = pytest.mark.exercise


def test_corrects_for_position():
    """Two products of EQUAL true quality (CTR=0.3) shown at different positions.
    The good position is examined fully (1.0); the bad one rarely (0.2). Raw rates
    differ; position-corrected rates should converge."""
    rng = np.random.default_rng(0)
    exam = {1: 1.0, 5: 0.2}
    rows = []
    for _ in range(4000):
        rows.append({"product_id": "A", "position": 1, "click": rng.binomial(1, 1.0 * 0.3)})
        rows.append({"product_id": "B", "position": 5, "click": rng.binomial(1, 0.2 * 0.3)})
    df = pd.DataFrame(rows)

    res = position_adjusted_rate(df, examination_by_position=exam).set_index("product_id")
    # naive rates differ a lot (B looks terrible only because it's buried)
    assert abs(res.loc["A", "naive_rate"] - res.loc["B", "naive_rate"]) > 0.1
    # corrected rates recover the equal true quality
    assert abs(res.loc["A", "corrected_rate"] - res.loc["B", "corrected_rate"]) < 0.06
