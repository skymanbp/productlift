"""Assemble the modeling matrix from the feature builders + label. MOSTLY BUILT.

This orchestrates the leakage-safe pipeline you implement in the other modules:
  1. split the enriched base table at `as_of` into a FEATURE window and an
     OUTCOME window (here is where leakage is prevented, structurally).
  2. build features from the feature window only.
  3. build the label from the outcome window only.
  4. join on product_id → the modeling matrix.

You implement `assemble`'s body once the feature/label functions exist; the wiring
and the windowing logic are provided so the leakage-safe structure is clear.
"""

from __future__ import annotations

import pandas as pd

from app.data.labels import label_underperforming
from app.features.behavioral import behavioral_features
from app.features.product import product_features
from app.features.temporal import temporal_features

FEATURE_COLS_TIME = "order_purchase_timestamp"


def split_windows(
    base: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FULLY BUILT — split events into feature (<= as_of) and outcome (as_of, as_of+h]."""
    t = base[FEATURE_COLS_TIME]
    feature_events = base[t <= as_of].copy()
    outcome_end = as_of + pd.Timedelta(days=horizon_days)
    outcome_events = base[(t > as_of) & (t <= outcome_end)].copy()
    return feature_events, outcome_events


def assemble(
    base: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int, **label_kwargs
) -> pd.DataFrame:
    """Return the product-level modeling matrix: features + is_underperforming.

    Two decisions here ARE the leakage discipline:

    - Only `is_underperforming` crosses the label join. The label frame also carries
      revenue/n_orders/category measured in the OUTCOME window; as features those
      would encode the target itself (and n_orders would shadow the feature-window
      behavioral column of the same name).
    - Inner join: a product enters the matrix only with BOTH a feature-window history
      and a judgeable outcome-window label. A product with no label was excluded by
      labels.py as unjudgeable (< min_orders) — that is not evidence of health, so we
      drop it rather than label it 0. The defensible alternative (treat vanishing
      from the outcome window as underperformance itself) is a different target
      definition and belongs in labels.py, not in a join default.
    """
    feature_events, outcome_events = split_windows(base, as_of, horizon_days)

    feats = (
        product_features(feature_events)
        .merge(behavioral_features(feature_events), on="product_id", validate="one_to_one")
        .merge(temporal_features(feature_events, as_of), on="product_id", validate="one_to_one")
    )
    labels = label_underperforming(outcome_events, **label_kwargs)

    return feats.merge(
        labels[["product_id", "is_underperforming"]],
        on="product_id",
        how="inner",
        validate="one_to_one",
    )
