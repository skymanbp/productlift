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

    TODO(lillian): group feature_events by product_id and compute at least:
      - n_orders (count), n_unique_customers (nunique on customer_id)
      - avg_price, avg_freight
      - avg_review_score, bad_review_rate (share of review_score <= 2)
      - late_delivery_rate, avg_delivery_days
    Handle products with no reviews (review_score all NaN) without crashing —
    decide on a sensible fill and document it. The test checks the bad_review_rate
    and the NaN handling.
    """
    raise NotImplementedError("Implement behavioral_features — see tests/test_features.py")
