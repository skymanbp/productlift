"""Tests for app/models (Milestone 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.baseline import build_baseline
from app.models.calibrate import reliability_table
from app.models.gbdt import build_gbdt, compute_scale_pos_weight, fit_gbdt

pytestmark = pytest.mark.exercise

TARGET = "is_underperforming"


def _xy(matrix):
    return matrix.drop(columns=[TARGET]), matrix[TARGET]


def test_baseline_pipeline_predicts_probabilities(modeling_matrix):
    X, y = _xy(modeling_matrix)
    numeric = ["x1", "x2"]
    categorical = ["product_category_name"]
    pipe = build_baseline(numeric, categorical)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert len(proba) == len(y)


def test_compute_scale_pos_weight():
    y = pd.Series([0] * 80 + [1] * 20)  # 80 neg / 20 pos
    assert compute_scale_pos_weight(y) == pytest.approx(4.0)


def test_gbdt_fits_and_predicts(modeling_matrix):
    X, y = _xy(modeling_matrix)
    X = X.drop(columns=["product_category_name"])  # GBDT test uses numeric-only for simplicity
    spw = compute_scale_pos_weight(y)
    model = build_gbdt(scale_pos_weight=spw)
    model = fit_gbdt(model, X, y, X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # signal exists in the fixture, so a fitted model should beat chance on train
    from evaluation.metrics import roc_auc

    assert roc_auc(y, proba) > 0.6


def test_reliability_table_on_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    table = reliability_table(y, p, n_bins=10)
    # observed frequency should track mean predicted prob within each bin
    arr = np.asarray(table)
    mean_pred = arr[:, 0]
    observed = arr[:, 1]
    assert np.nanmax(np.abs(mean_pred - observed)) < 0.05
