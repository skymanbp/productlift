"""Drift monitor: training-window features vs the latest snapshot.

M7 built PSI and drift_report but nothing called them. This rebuilds the
feature snapshot at two as-of dates from the same base table:
  reference = features as of the training cutoff (what the model learned from)
  current   = features as of the latest event in the data (what serving sees)
and compares them per feature. Exit code 0 = calm, 1 = breach, so a scheduled
run can act as a retraining trigger wired to alerting rather than a dashboard.

    python -m scripts.monitor [--fail-on significant|moderate]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from app.config import get_params, get_settings
from app.features.behavioral import behavioral_features
from app.features.product import product_features
from app.features.temporal import temporal_features
from observability.monitoring import drift_report

TIME_COL = "order_purchase_timestamp"

# Which statuses each --fail-on level trips on. no_data always trips: a feature
# with zero observed values is a broken pipeline, not a quiet day.
FAIL_LEVELS = {"moderate": {"moderate", "significant", "no_data"},
               "significant": {"significant", "no_data"}}


def feature_snapshot(base: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Product-level features as of a date: the same three builders and joins as
    app/features/build.py::assemble, without the label (there are no labels when
    monitoring inputs)."""
    events = base[base[TIME_COL] <= as_of].copy()
    return (
        product_features(events)
        .merge(behavioral_features(events), on="product_id", validate="one_to_one")
        .merge(temporal_features(events, as_of), on="product_id", validate="one_to_one")
    )


def breached(report: pd.DataFrame, fail_on: str) -> bool:
    """True if any feature's status trips the fail level."""
    if fail_on not in FAIL_LEVELS:
        raise ValueError(f"fail_on must be one of {sorted(FAIL_LEVELS)}, got {fail_on!r}")
    return bool(report["status"].isin(FAIL_LEVELS[fail_on]).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on",
        choices=sorted(FAIL_LEVELS),
        default="significant",
        help="drift level that makes the exit code non-zero (default: significant)",
    )
    args = parser.parse_args()

    settings = get_settings()
    params = get_params()
    base = pd.read_parquet(settings.interim_dir / "base_table.parquet")

    ref_as_of = pd.Timestamp(params["split"]["train_end"])
    cur_as_of = base[TIME_COL].max()
    reference = feature_snapshot(base, ref_as_of)
    current = feature_snapshot(base, cur_as_of)

    features = sorted(
        c for c in reference.select_dtypes("number").columns if c in current.columns
    )
    report = drift_report(reference, current, features)

    print(f"Drift report: reference as of {ref_as_of.date()} "
          f"({len(reference)} products) vs current as of {cur_as_of.date()} "
          f"({len(current)} products)\n")
    print(report.to_string(index=False))

    if breached(report, args.fail_on):
        n = int(report["status"].isin(FAIL_LEVELS[args.fail_on]).sum())
        print(f"\nBREACH: {n} feature(s) at/above '{args.fail_on}'; investigate or retrain.")
        sys.exit(1)
    print(f"\nOK: no feature at/above '{args.fail_on}'.")


if __name__ == "__main__":
    main()
