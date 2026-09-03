"""Tests for app/features/build.py::assemble (Milestone 2 integration).

Depends on the feature + label exercises being done — it's the milestone that brings
the data pipeline together.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.features.build import assemble, split_windows

pytestmark = pytest.mark.exercise


def test_split_windows_is_leakage_safe(base_events, as_of):
    # split_windows is provided/built; this documents the contract the rest relies on.
    feat, outcome = split_windows(base_events, as_of, horizon_days=180)
    assert feat["order_purchase_timestamp"].max() <= as_of
    assert outcome["order_purchase_timestamp"].min() > as_of


def test_assemble_produces_product_level_matrix(base_events, as_of):
    matrix = assemble(base_events, as_of=as_of, horizon_days=180, quantile=0.5, min_orders=3)
    # one row per product, with the label and at least one engineered feature
    assert matrix["product_id"].is_unique
    assert "is_underperforming" in matrix.columns
    assert set(matrix["is_underperforming"].unique()) <= {0, 1}
    assert "n_orders" in matrix.columns  # a behavioral feature made it through


def test_average_sale_price_appears_once(base_events, as_of):
    # Regression: product.avg_unit_price and behavioral.avg_price were the same
    # aggregation — mean of `price` per product over the same feature window — so
    # the matrix carried one number under two names. Exactly one may survive.
    feature_events, _ = split_windows(base_events, as_of, horizon_days=180)
    mean_price = feature_events.groupby("product_id")["price"].mean()

    matrix = assemble(
        base_events, as_of=as_of, horizon_days=180, quantile=0.5, min_orders=3
    ).set_index("product_id")
    expected = mean_price.reindex(matrix.index)

    duplicates = [
        col
        for col in matrix.select_dtypes("number").columns
        if np.allclose(matrix[col], expected)
    ]
    assert duplicates == ["avg_unit_price"]
