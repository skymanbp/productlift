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
    """Return a product-level frame [product_id, revenue, n_orders, is_underperforming].

    `outcome_events` is the order-item table restricted to the OUTCOME window (rows
    after the as-of date). Do NOT pass the full history here — that's the caller's
    job, and the separation is what prevents leakage.

    TODO(lillian):
      1. aggregate to product level: revenue = sum(price), n_orders = row count,
         keep the product's category.
      2. drop products with n_orders < min_orders (too little signal to judge).
      3. within each category, compute the `quantile` revenue threshold.
      4. is_underperforming = 1 if the product's revenue <= its category threshold,
         else 0. Return the product-level frame.
    Edge cases the test checks: a category with one product, and the min_orders cut.
    """
    raise NotImplementedError("Implement label_underperforming — see tests/test_labels.py")
