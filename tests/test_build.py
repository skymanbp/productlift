"""Tests for app/features/build.py::assemble (Milestone 2 integration).

Depends on the feature + label exercises being done — it's the milestone that brings
the data pipeline together.
"""

from __future__ import annotations

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
