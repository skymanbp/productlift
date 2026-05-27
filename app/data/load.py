"""Load raw Olist CSVs and assemble the analytic base table. FULLY BUILT.

This is plumbing — joins and type parsing, not ML. Read it to understand the data
model, but you don't implement anything here. The interesting (and graded) work is
labeling (labels.py), splitting (splits.py), and features/.

THE ANALYTIC BASE TABLE
    We produce one row per *order item* (a product within an order), enriched with:
      - product attributes (category, photos, description length, dimensions)
      - order facts (timestamps, status, price, freight)
      - the review score for that order (our satisfaction signal)
    Downstream, features/ aggregate this to the product level (using only data
    BEFORE the as-of cutoff) and labels.py defines the target (using data AFTER it).

OLIST SCHEMA: see docs/data-dictionary.md for every column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The Olist CSV filenames (as distributed on Kaggle).
FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items": ["shipping_limit_date"],
}


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every Olist CSV into a dict of DataFrames, parsing dates."""
    out: dict[str, pd.DataFrame] = {}
    for key, fname in FILES.items():
        path = Path(raw_dir) / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `make data` (scripts/download_data.py) first — "
                "see data/README.md."
            )
        df = pd.read_csv(path)
        for col in DATE_COLS.get(key, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        out[key] = df
    return out


def build_base_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join items → orders → products → reviews into one enriched event table.

    One row per order item. This is the input to both labeling and feature building.
    We keep only delivered orders (the lifecycle is complete and the review is
    meaningful) — an explicit, documented modeling choice.
    """
    items = raw["items"]
    orders = raw["orders"]
    products = raw["products"]
    reviews = raw["reviews"]
    cats = raw["category_translation"]

    df = items.merge(orders, on="order_id", how="inner", validate="many_to_one")
    df = df.merge(products, on="product_id", how="left", validate="many_to_one")

    # One review score per order (take the first if duplicated).
    review_score = (
        reviews.sort_values("review_creation_date")
        .groupby("order_id", as_index=False)["review_score"]
        .first()
    )
    df = df.merge(review_score, on="order_id", how="left", validate="many_to_one")

    # English category names are friendlier for EDA / interpretation.
    df = df.merge(cats, on="product_category_name", how="left")

    df = df[df["order_status"] == "delivered"].copy()

    # A couple of always-useful derived fields (knowable at/after delivery).
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    df["late_delivery"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype("Int64")

    return df.reset_index(drop=True)
