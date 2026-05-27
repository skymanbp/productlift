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


def assemble(base: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int, **label_kwargs) -> pd.DataFrame:
    """Return the product-level modeling matrix: features + is_underperforming.

    TODO(lillian) — once the builders are implemented, wire them up:
      1. feature_events, outcome_events = split_windows(base, as_of, horizon_days)
      2. feats = product_features(feature_events)
              .merge(behavioral_features(feature_events), on="product_id")
              .merge(temporal_features(feature_events, as_of), on="product_id")
      3. labels = label_underperforming(outcome_events, **label_kwargs)
      4. join feats to labels on product_id. Decide the join type and document why
         (a product with features but no outcome-window orders has no label — drop
         it, or is that itself the signal? this is a real modeling decision to make
         and defend).
    The test builds a tiny base table and checks no outcome-window column leaks into
    the feature columns.
    """
    raise NotImplementedError("Implement assemble — see tests/test_build.py")
