"""Shared test fixtures.

The whole suite runs on tiny in-memory frames — never the real Olist download — so
it's fast and deterministic. Fixtures mimic the schema of build_base_table's output
(one row per order item) so feature/label tests exercise real code paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(42)


def _event(
    product_id, order_id, customer_id, price, ts, review_score, category,
    *, photos=3, desc_len=200, name_len=40, weight=500, length=20, height=10, width=15,
    freight=15.0, late=0, delivery_days=10.0,
) -> dict:
    return {
        "product_id": product_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "price": price,
        "freight_value": freight,
        "order_purchase_timestamp": pd.Timestamp(ts),
        "review_score": review_score,
        "product_category_name": category,
        "product_photos_qty": photos,
        "product_description_lenght": desc_len,
        "product_name_lenght": name_len,
        "product_weight_g": weight,
        "product_length_cm": length,
        "product_height_cm": height,
        "product_width_cm": width,
        "late_delivery": late,
        "delivery_days": delivery_days,
    }


@pytest.fixture
def base_events() -> pd.DataFrame:
    """A small enriched order-item table spanning before/after the 2018-03-31 cutoff,
    across two categories, with a clear underperformer per category."""
    rows: list[dict] = []
    oid = 0

    def add(pid, price, ts, score, cat, **kw):
        nonlocal oid
        oid += 1
        rows.append(_event(pid, f"o{oid}", f"c{oid % 7}", price, ts, score, cat, **kw))

    # Category A — alpha sells a lot (healthy), gamma barely sells (underperformer)
    for d in range(1, 13):
        add("alpha", 100.0, f"2018-0{1 + d % 3}-{(d % 27) + 1:02d}", 5, "cat_a")
    for d in range(1, 7):
        add("beta", 80.0, f"2018-02-{(d % 27) + 1:02d}", 4, "cat_a")
    for d in range(1, 6):
        add("gamma", 20.0, f"2018-01-{(d % 27) + 1:02d}", 2, "cat_a", late=1)

    # Category B
    for d in range(1, 11):
        add("delta", 60.0, f"2018-03-{(d % 27) + 1:02d}", 5, "cat_b")
    for d in range(1, 6):
        add("epsilon", 15.0, f"2018-02-{(d % 27) + 1:02d}", 3, "cat_b")

    # Post-cutoff orders (the OUTCOME window) — used for labels, never features.
    for d in range(1, 9):
        add("alpha", 100.0, f"2018-05-{(d % 27) + 1:02d}", 5, "cat_a")
    for d in range(1, 6):
        add("gamma", 20.0, f"2018-05-{(d % 27) + 1:02d}", 2, "cat_a")
    for d in range(1, 8):
        add("delta", 60.0, f"2018-06-{(d % 27) + 1:02d}", 5, "cat_b")
    for d in range(1, 6):
        add("epsilon", 15.0, f"2018-06-{(d % 27) + 1:02d}", 3, "cat_b")

    return pd.DataFrame(rows)


@pytest.fixture
def as_of() -> pd.Timestamp:
    return pd.Timestamp("2018-03-31")


@pytest.fixture
def modeling_matrix() -> pd.DataFrame:
    """A synthetic product-level matrix for model tests: numeric + categorical
    features with an imbalanced binary target that the features actually predict."""
    n = 400
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    cat = RNG.choice(["a", "b", "c"], size=n)
    # signal: high x1 and category 'c' -> more likely underperforming
    logit = 1.5 * x1 - 0.5 * x2 + (cat == "c") * 1.0 - 1.6
    p = 1 / (1 + np.exp(-logit))
    y = (RNG.uniform(size=n) < p).astype(int)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "product_category_name": cat, "is_underperforming": y}
    )
