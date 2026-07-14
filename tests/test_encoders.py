"""Tests for app/features/encoders.py (Milestone 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.encoders import SmoothedTargetEncoder, oof_target_encode

pytestmark = pytest.mark.exercise


@pytest.fixture
def data():
    # 'a' frequent (mean 0.8), 'b' rare (single 1.0)
    cats = ["a"] * 20 + ["b"]
    y = pd.Series([1] * 16 + [0] * 4 + [1])
    X = pd.DataFrame({"cat": cats})
    return X, y


def test_frequent_category_near_raw_mean(data):
    X, y = data
    enc = SmoothedTargetEncoder("cat", smoothing=20.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"cat": ["a"]})).ravel()[0]
    assert out == pytest.approx(0.80, abs=0.03)  # large n -> little shrinkage


def test_rare_category_shrinks_to_global(data):
    X, y = data
    global_mean = y.mean()
    enc = SmoothedTargetEncoder("cat", smoothing=20.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"cat": ["b"]})).ravel()[0]
    # pulled away from its raw mean (1.0) toward the global mean
    assert abs(out - global_mean) < abs(1.0 - global_mean)


def test_unseen_category_gets_global_mean(data):
    X, y = data
    enc = SmoothedTargetEncoder("cat", smoothing=20.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"cat": ["never_seen"]})).ravel()[0]
    assert out == pytest.approx(y.mean(), abs=1e-9)


def test_output_is_2d(data):
    X, y = data
    enc = SmoothedTargetEncoder("cat").fit(X, y)
    out = enc.transform(X)
    assert np.asarray(out).ndim == 2  # composes in a ColumnTransformer


# --- OOF encoding ----------------------------------------------------------


def test_oof_own_label_never_reaches_own_feature():
    # Every label is 0 except one row, so any positive encoding for that row can
    # only come from its own label. OOF must give it exactly 0; naive
    # fit-then-transform on train leaks and gives > 0.
    n = 40
    X = pd.DataFrame({"cat": ["a"] * n})
    y = pd.Series([0] * n)
    y.iloc[7] = 1

    naive = SmoothedTargetEncoder("cat", smoothing=20.0).fit(X, y).transform(X).ravel()
    oof = oof_target_encode(X, y, "cat", smoothing=20.0, n_splits=5, seed=0).ravel()

    assert naive[7] > 0.0  # own label leaked in
    assert oof[7] == 0.0  # own label excluded


def test_oof_shape_and_determinism(data):
    X, y = data
    a = oof_target_encode(X, y, "cat", n_splits=3, seed=42)
    b = oof_target_encode(X, y, "cat", n_splits=3, seed=42)
    assert a.shape == (len(X), 1)
    assert np.array_equal(a, b)  # same seed -> same folds -> same encoding


def test_oof_rejects_bad_inputs(data):
    X, y = data
    with pytest.raises(ValueError, match="n_splits"):
        oof_target_encode(X, y, "cat", n_splits=1)
    with pytest.raises(ValueError, match="rows"):
        oof_target_encode(X, y.iloc[:-1], "cat")
