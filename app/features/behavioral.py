"""Behavioral / interaction features.  ⛏️  EXERCISE — Milestone 2.

LEARNING GOAL
    Aggregate a product's order history into signals of demand and satisfaction.
    These are the strongest predictors — and the easiest to leak. Every aggregate
    here must use ONLY the feature window (pre as-of); the caller guarantees the
    input is already filtered, but you must not pull in anything outcome-derived.

KEY CONCEPT — aggregation + the cold-start tradeoff
    Per product, over the feature window:
      - n_orders, n_unique_customers (demand/popularity)
      - avg/median price actually sold at, avg freight_value
      - avg review_score so far, share of bad reviews (<=2)   [satisfaction]
      - late_delivery rate, avg delivery_days                 [fulfillment]
    More history = more reliable aggregates, but new products have little. Note
    where you'd apply Bayesian smoothing toward the category mean (ties to
    encoders.py) — interviewers love this.

WATCH OUT
    review_score and delivery outcomes are fine as FEATURES only because they're from
    PAST orders in the feature window. The label uses FUTURE orders. Keep that line
    crisp in your head — it's the whole leakage lesson.

TEST
    tests/test_features.py::test_behavioral_features
"""

from __future__ import annotations

import pandas as pd


def behavioral_features(feature_events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per product of historical behavioral aggregates.

    All aggregates use only the feature window the caller passes in. Rates are
    computed among the orders that carry the signal: `bad_review_rate` is the share
    of REVIEWED orders scoring <= 2 (counting unreviewed orders as "not bad" would
    bias every sparse product toward looking healthy). Products with no reviews get
    NaN for both review aggregates — deliberately not filled here: missingness is
    itself signal, and any imputation must be fit on train only, which is the model
    pipeline's job (M3), not the feature builder's.
    """
    required = {
        "product_id",
        "order_id",
        "customer_id",
        "price",
        "freight_value",
        "review_score",
        "late_delivery",
        "delivery_days",
    }
    missing = required - set(feature_events.columns)
    if missing:
        raise ValueError(f"feature_events is missing columns: {sorted(missing)}")

    return feature_events.groupby("product_id", as_index=False).agg(
        n_orders=("order_id", "size"),
        n_unique_customers=("customer_id", "nunique"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_review_score=("review_score", "mean"),
        # mean() of an empty (all-NaN-dropped) slice is NaN, not an error.
        bad_review_rate=("review_score", lambda s: s.dropna().le(2).mean()),
        late_delivery_rate=("late_delivery", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
    )
