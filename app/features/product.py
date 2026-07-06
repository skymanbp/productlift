"""Product (static) features.  ⛏️  EXERCISE — Milestone 2.

LEARNING GOAL
    Engineer features from a product's intrinsic attributes — the things true of the
    listing itself, independent of time. These are available even for near-cold
    products, which is exactly why they matter for the cold-start problem the
    interview doc calls out.

KEY CONCEPT — content features & sensible transforms
    Raw Olist product columns (photos count, description length, weight, dimensions)
    are weak alone but predictive together. Engineer:
      - listing-quality signals: n photos, description length, name length
      - physical: weight, volume (l*w*h), density
      - price level (avg unit price) and price relative to category median
    Watch for skew — physical sizes and prices are heavy-tailed; a log transform
    is usually the right move (note it for the modeling step).

INTERVIEW ANGLE
    "What features would you engineer?" → lead with content/listing-quality features
    because they solve cold start, then explain the category-relative price as a way
    to compare like with like.

TEST
    tests/test_features.py::test_product_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = {
    "product_id",
    "product_category_name",
    "price",
    "product_photos_qty",
    "product_description_lenght",  # Olist's own typo — keep the raw name as-is
    "product_name_lenght",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
}


def product_features(feature_events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per product of static/listing features.

    `feature_events` is the order-item table restricted to the FEATURE window.
    Static attributes are per-product constants (taken with "first"); NaN attributes
    stay NaN — missingness is signal and imputation belongs to the model pipeline,
    where it can be fit on train only. `price_vs_category_median` compares each
    product's average sale price to the median of its category peers' averages,
    all computed inside the feature window: X-only statistics, no label, nothing
    after as_of. The raw category column is kept for the M3 target encoder.
    """
    missing = REQUIRED_COLS - set(feature_events.columns)
    if missing:
        raise ValueError(f"feature_events is missing columns: {sorted(missing)}")

    feats = feature_events.groupby("product_id", as_index=False).agg(
        product_category_name=("product_category_name", "first"),
        n_photos=("product_photos_qty", "first"),
        description_length=("product_description_lenght", "first"),
        name_length=("product_name_lenght", "first"),
        weight_g=("product_weight_g", "first"),
        length_cm=("product_length_cm", "first"),
        height_cm=("product_height_cm", "first"),
        width_cm=("product_width_cm", "first"),
        avg_unit_price=("price", "mean"),
    )
    feats["volume_cm3"] = feats["length_cm"] * feats["height_cm"] * feats["width_cm"]
    # where() turns zero/negative volumes into NaN instead of an infinite density.
    feats["density_g_cm3"] = feats["weight_g"] / feats["volume_cm3"].where(feats["volume_cm3"] > 0)
    # log1p tames the heavy right tails (EDA: price p99/p50 ≈ 12x; sizes similar).
    feats["log_weight_g"] = np.log1p(feats["weight_g"])
    feats["log_volume_cm3"] = np.log1p(feats["volume_cm3"])
    feats["log_avg_unit_price"] = np.log1p(feats["avg_unit_price"])

    category_median_price = feats.groupby("product_category_name")["avg_unit_price"].transform(
        "median"
    )
    feats["price_vs_category_median"] = feats["avg_unit_price"] / category_median_price

    # Raw dimensions are intermediates; volume + density carry the physical signal.
    return feats.drop(columns=["length_cm", "height_cm", "width_cm"])
