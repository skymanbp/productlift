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

import pandas as pd


def product_features(feature_events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per product of static/listing features.

    `feature_events` is the order-item table restricted to the FEATURE window.

    TODO(lillian): build at least:
      - product_id (key)
      - n_photos, description_length, name_length  (listing quality)
      - weight_g, volume_cm3 (= length*height*width), and a log version of skewed ones
      - avg_unit_price (mean price) and price_vs_category_median (ratio)
    Return product-level, one row per product_id. The test checks the volume and the
    category-relative price computation on a small fixture.
    """
    raise NotImplementedError("Implement product_features — see tests/test_features.py")
