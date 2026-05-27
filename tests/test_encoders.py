"""Tests for app/features/encoders.py — SmoothedTargetEncoder (Milestone 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.encoders import SmoothedTargetEncoder

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
