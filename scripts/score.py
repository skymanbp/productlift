"""Batch-score every product and write the top-k intervention list.

Training, evaluation and single-row serving already exist; this is the ranked
CSV a category manager acts on: every product scored by risk, plus the top
slice worth intervening on (repricing, content fixes, delisting review). The
top fraction defaults to config's evaluation.recall_k_fraction, so the list
size matches the operating point the offline report quotes recall@k for.

    python -m scripts.score [--top-fraction 0.10]
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from app.config import get_params, get_settings
from app.models.registry import ModelCard, load_model

ID_COL = "product_id"

# Extra columns shown next to the score so a reviewer can act on the row.
CONTEXT_COLS = [
    "product_category_name",
    "avg_unit_price",
    "price_vs_category_median",
    "n_orders",
    "days_since_last",
]


def rank_products(matrix: pd.DataFrame, model: Any, card: ModelCard) -> pd.DataFrame:
    """Score every product; return a frame ranked by risk, highest first.

    Selects features by card.feature_names (the train/serve contract), not by
    whatever columns happen to be present, which would break silently if the
    matrix gained a column the model never saw.
    """
    missing = [c for c in card.feature_names if c not in matrix.columns]
    if missing:
        raise ValueError(f"matrix is missing model features: {missing}")

    scores = model.predict_proba(matrix[card.feature_names])[:, 1]
    out = matrix[[ID_COL, *[c for c in CONTEXT_COLS if c in matrix.columns]]].copy()
    out["risk_score"] = scores
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=None,
        help="fraction of products for the intervention list (default: config recall_k_fraction)",
    )
    args = parser.parse_args()

    settings = get_settings()
    params = get_params()
    top_fraction = (
        args.top_fraction
        if args.top_fraction is not None
        else float(params["evaluation"]["recall_k_fraction"])
    )
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top-fraction must be in (0, 1], got {top_fraction}")

    matrix = pd.read_parquet(settings.processed_dir / "modeling_matrix.parquet")
    model, card = load_model(settings.model_path)

    ranked = rank_products(matrix, model, card)
    k = max(1, round(len(ranked) * top_fraction))
    top = ranked.head(k)

    all_path = settings.processed_dir / "product_scores.csv"
    top_path = settings.processed_dir / "intervention_list.csv"
    ranked.to_csv(all_path, index=False)
    top.to_csv(top_path, index=False)

    print(f"Scored {len(ranked)} products with {card.name}")
    print(f"Full ranking -> {all_path}")
    print(f"Top {k} ({top_fraction:.0%}) intervention list -> {top_path}")
    print("\nTop 10 preview:")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
