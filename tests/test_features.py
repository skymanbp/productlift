"""Tests for app/features (Milestone 2)."""

from __future__ import annotations

import pytest

from app.features.behavioral import behavioral_features
from app.features.product import product_features
from app.features.temporal import temporal_features

pytestmark = pytest.mark.exercise


def _feature_window(base_events, as_of):
    return base_events[base_events["order_purchase_timestamp"] <= as_of].copy()


def test_product_features(base_events, as_of):
    feats = product_features(_feature_window(base_events, as_of)).set_index("product_id")
    # volume = length * height * width = 20 * 10 * 15 = 3000
    assert feats.loc["alpha", "volume_cm3"] == pytest.approx(3000.0)
    # alpha (price 100) is above the cat_a median; gamma (price 20) below
    assert feats.loc["alpha", "price_vs_category_median"] > 1.0
    assert feats.loc["gamma", "price_vs_category_median"] < 1.0


def test_behavioral_features(base_events, as_of):
    feats = behavioral_features(_feature_window(base_events, as_of)).set_index("product_id")
    # gamma's reviews are all score 2 -> bad_review_rate = 1.0; alpha all 5 -> 0.0
    assert feats.loc["gamma", "bad_review_rate"] == pytest.approx(1.0)
    assert feats.loc["alpha", "bad_review_rate"] == pytest.approx(0.0)
    # alpha has 12 feature-window orders
    assert feats.loc["alpha", "n_orders"] == 12
    # gamma's deliveries are all late
    assert feats.loc["gamma", "late_delivery_rate"] == pytest.approx(1.0)


def test_temporal_features(base_events, as_of):
    feats = temporal_features(_feature_window(base_events, as_of), as_of).set_index("product_id")
    assert (feats["tenure_days"] >= 0).all()
    assert (feats["days_since_last"] >= 0).all()
    # window counts can't exceed total orders
    assert feats.loc["alpha", "orders_last_90d"] >= 1


def test_temporal_features_guards_against_leakage(base_events, as_of):
    # Passing events AFTER as_of must be rejected — the cheap guard that stops
    # look-ahead leakage at the source.
    with pytest.raises((ValueError, AssertionError)):
        temporal_features(base_events, as_of)  # base_events includes post-cutoff rows
