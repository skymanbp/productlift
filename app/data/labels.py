"""Define the prediction target: which products are "underperforming".  ⛏️  EXERCISE — Milestone 1.

LEARNING GOAL
    Turn a fuzzy business notion ("underperforming products") into a precise,
    defensible label. How you define the target IS the modeling problem — get this
    wrong and nothing downstream matters. This is Step 1–2 of the interview
    framework (problem framing) made concrete.

KEY CONCEPT — relative, within-group labels
    Raw revenue is dominated by category (electronics >> stationery), so a global
    threshold just relabels "category". We define underperformance RELATIVE TO THE
    PRODUCT'S CATEGORY: a product is underperforming if its revenue is in the bottom
    `quantile` of its category, among products with enough orders to judge.

    Just as important — and the reason this is in labels.py, not features.py — the
    label must be computed on a DIFFERENT time window than the features (the
    "outcome" window, AFTER the as-of date). Mixing them is target leakage.

INTERVIEW ANGLE
    Expect: "How did you define underperformance, and why?" Have the within-category
    quantile rationale and the min-orders cutoff ready, plus the tradeoff you chose.

TEST
    tests/test_labels.py
"""

from __future__ import annotations

import pandas as pd


def label_underperforming(
    outcome_events: pd.DataFrame,
    *,
    category_col: str = "product_category_name",
    product_col: str = "product_id",
    price_col: str = "price",
    quantile: float = 0.25,
    min_orders: int = 5,
) -> pd.DataFrame:
    """Return a product-level frame [product_id, category, revenue, n_orders, is_underperforming].

    `outcome_events` is the order-item table restricted to the OUTCOME window (rows
    after the as-of date). Do NOT pass the full history here — that's the caller's
    job, and the separation is what prevents leakage.

    The threshold is a per-category revenue quantile computed AMONG the products
    that survive the min_orders cut — sparse products should neither be judged nor
    drag the bar down for everyone else. A single-product category labels its only
    product 1 (its revenue equals its own quantile, and `<=` includes the boundary):
    with no peers there is no evidence it performs, which matches the recall-first
    business goal of surfacing candidates for review. Products with NaN category
    are dropped by the groupby: "underperforming" is only defined relative to a
    peer group, and there is none to compare against.
    """
    required = {product_col, category_col, price_col}
    missing = required - set(outcome_events.columns)
    if missing:
        raise ValueError(f"outcome_events is missing columns: {sorted(missing)}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    products = outcome_events.groupby([product_col, category_col], as_index=False).agg(
        revenue=(price_col, "sum"), n_orders=(price_col, "size")
    )
    products = products[products["n_orders"] >= min_orders].reset_index(drop=True)

    threshold = products.groupby(category_col)["revenue"].transform(lambda s: s.quantile(quantile))
    products["is_underperforming"] = (products["revenue"] <= threshold).astype(int)
    return products
