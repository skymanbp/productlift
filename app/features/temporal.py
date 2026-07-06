"""Temporal features.  ⛏️  EXERCISE — Milestone 2.

LEARNING GOAL
    Capture recency, trend, and tenure — when a product sold, not just how much.
    Temporal features are where look-ahead leakage sneaks in, so they're the best
    place to internalize "as-of" thinking.

KEY CONCEPT — everything is relative to the as-of date
    Given `as_of` (the prediction timestamp), per product compute:
      - tenure_days        : as_of - first_order_date   (how established)
      - days_since_last    : as_of - last_order_date     (recency)
      - orders_last_30/90d : count in the trailing window before as_of (momentum)
      - trend              : orders in recent window vs earlier window (accelerating?)
    NONE of these may reference any timestamp after as_of.

INTERVIEW ANGLE
    Recency/tenure features are how you handle the cold-start vs established-product
    distinction, and how you'd detect a product losing momentum before revenue
    fully reflects it.

TEST
    tests/test_features.py::test_temporal_features
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def temporal_features(
    feature_events: pd.DataFrame,
    as_of: pd.Timestamp,
    time_col: str = "order_purchase_timestamp",
    windows: Sequence[int] = (7, 30, 90),
    trend_window: int = 30,
) -> pd.DataFrame:
    """Return one row per product of time-based features measured at `as_of`.

    The first thing this function does is refuse events after `as_of`: every
    downstream feature is only meaningful if "now" really is as_of, so look-ahead
    rows are a hard error, not something to filter quietly (silently dropping them
    would hide the caller's windowing bug). `windows` defaults mirror
    config/params.yaml features.rolling_windows_days. `trend_{w}d` is the momentum
    ratio orders(last w days) / orders(the w days before that); NaN when the product
    had no orders in the prior window — acceleration is undefined without a base,
    and 0-vs-inf fills would fabricate signal.
    """
    t = feature_events[time_col]
    n_future = int((t > as_of).sum())
    if n_future:
        raise ValueError(
            f"feature_events contains {n_future} rows after as_of={as_of} — "
            "look-ahead leakage; fix the caller's windowing"
        )

    per_product = feature_events.groupby("product_id", as_index=False).agg(
        first_order=(time_col, "min"), last_order=(time_col, "max")
    )
    feats = pd.DataFrame({"product_id": per_product["product_id"]})
    feats["tenure_days"] = (as_of - per_product["first_order"]).dt.total_seconds() / 86400.0
    feats["days_since_last"] = (as_of - per_product["last_order"]).dt.total_seconds() / 86400.0

    for w in windows:
        in_window = feature_events[t > as_of - pd.Timedelta(days=w)]
        counts = in_window.groupby("product_id").size()
        feats[f"orders_last_{w}d"] = feats["product_id"].map(counts).fillna(0).astype(int)

    recent_start = as_of - pd.Timedelta(days=trend_window)
    prior_start = as_of - pd.Timedelta(days=2 * trend_window)
    recent = feature_events[t > recent_start].groupby("product_id").size()
    prior = feature_events[(t > prior_start) & (t <= recent_start)].groupby("product_id").size()
    feats[f"trend_{trend_window}d"] = feats["product_id"].map(recent).fillna(0) / feats[
        "product_id"
    ].map(prior)

    return feats
