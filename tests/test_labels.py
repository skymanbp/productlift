"""Tests for app/data/labels.py (Milestone 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.labels import label_underperforming

pytestmark = pytest.mark.exercise


def test_within_category_bottom_quantile(base_events):
    # Use the post-cutoff window as the outcome window.
    outcome = base_events[base_events["order_purchase_timestamp"] > pd.Timestamp("2018-03-31")]
    labels = label_underperforming(outcome, quantile=0.5, min_orders=3)

    labels = labels.set_index("product_id")
    # In cat_a, gamma (low revenue) should be flagged, alpha (high) should not.
    assert labels.loc["gamma", "is_underperforming"] == 1
    assert labels.loc["alpha", "is_underperforming"] == 0
    # label is binary
    assert set(labels["is_underperforming"].unique()) <= {0, 1}


def test_min_orders_filter_excludes_sparse_products(base_events):
    outcome = base_events[base_events["order_purchase_timestamp"] > pd.Timestamp("2018-03-31")]
    labels = label_underperforming(outcome, quantile=0.25, min_orders=100)
    # nobody has 100+ orders in the tiny fixture -> empty label set
    assert len(labels) == 0
