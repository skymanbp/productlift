"""Build the modeling matrix from raw Olist. MOSTLY BUILT (orchestration).

Ties the Milestone 1–2 data and feature work into one runnable pipeline: load raw
→ base table → temporal split → features + label → joined matrix →
data/processed/modeling_matrix.parquet. It runs end to end.

    python -m scripts.build_features
"""

from __future__ import annotations

import pandas as pd

from app.config import get_params, get_settings
from app.data.load import build_base_table, load_raw
from app.features.build import assemble


def main() -> None:
    settings = get_settings()
    params = get_params()

    raw = load_raw(settings.raw_dir)
    base = build_base_table(raw)

    # Save the enriched base table (useful for EDA in the notebook).
    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    base.to_parquet(settings.interim_dir / "base_table.parquet", index=False)
    print(f"Base table: {len(base):,} order-item rows -> data/interim/base_table.parquet")

    # As-of cutoff = end of the train window; horizon = the rest, capped sensibly.
    as_of = pd.Timestamp(params["split"]["train_end"])
    horizon_days = 180
    matrix = assemble(
        base,
        as_of=as_of,
        horizon_days=horizon_days,
        quantile=params["target"]["underperformance_quantile"],
        min_orders=params["target"]["min_orders_for_label"],
    )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.processed_dir / "modeling_matrix.parquet"
    matrix.to_parquet(out, index=False)
    print(f"Modeling matrix: {len(matrix):,} products x {matrix.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
