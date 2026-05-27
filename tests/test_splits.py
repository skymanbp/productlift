"""Tests for app/data/splits.py (Milestone 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.splits import temporal_split

pytestmark = pytest.mark.exercise


def test_temporal_split_partitions_by_time(base_events):
    s = temporal_split(base_events, "order_purchase_timestamp", "2018-03-31", "2018-05-31")
    # windows are non-overlapping and exhaustive
    assert len(s.train) + len(s.valid) + len(s.test) == len(base_events)
    # NO temporal leakage across the boundary — the whole point
    assert s.train["order_purchase_timestamp"].max() <= pd.Timestamp("2018-03-31")
    assert s.valid["order_purchase_timestamp"].min() > pd.Timestamp("2018-03-31")
    assert s.test["order_purchase_timestamp"].min() > pd.Timestamp("2018-05-31")


def test_no_row_appears_twice(base_events):
    s = temporal_split(base_events, "order_purchase_timestamp", "2018-03-31", "2018-05-31")
    ids = pd.concat([s.train["order_id"], s.valid["order_id"], s.test["order_id"]])
    assert ids.is_unique
