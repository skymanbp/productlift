"""Train and register a model. MOSTLY BUILT (orchestration).

Runs once Milestones 1–3 are implemented. Shows the canonical flow:
baseline first, then GBDT, then calibrate, then register with a model card.

    python -m scripts.train
"""

from __future__ import annotations

import pandas as pd

from app.config import get_params, get_settings
from app.models.baseline import build_baseline, fit_predict_proba
from app.models.calibrate import calibrate
from app.models.gbdt import build_gbdt, compute_scale_pos_weight, fit_gbdt
from app.models.registry import ModelCard, save_model
from evaluation.metrics import pr_auc

TARGET = "is_underperforming"


def main() -> None:
    settings = get_settings()
    params = get_params()
    matrix = pd.read_parquet(settings.processed_dir / "modeling_matrix.parquet")

    # NOTE: for the product-level matrix we split products; you decide the split key
    # and strategy (this is part of the M3 exercise — a held-out test set you only
    # touch once). A simple seeded random split over products is acceptable here
    # since the temporal leakage was already handled at feature/label time.
    y = matrix[TARGET]
    X = matrix.drop(columns=[TARGET])
    feature_names = list(X.columns)

    # --- Baseline first (always) ---
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    baseline = build_baseline(numeric, categorical)
    _ = fit_predict_proba(baseline, X, y, X)  # sanity: it fits
    print("Baseline pipeline fit OK.")

    # --- GBDT with imbalance handling ---
    spw = compute_scale_pos_weight(y) if params["model"]["handle_imbalance"] == "scale_pos_weight" else None
    gbdt = build_gbdt(scale_pos_weight=spw)
    gbdt = fit_gbdt(gbdt, X, y, X, y)  # replace with real train/valid in your impl
    model = calibrate(gbdt, X, y) if params["model"]["calibrate"] else gbdt

    card = ModelCard(
        name="productlift_gbdt",
        params={"model": params["model"], "tuning": params["tuning"]},
        metrics={"train_pr_auc": pr_auc(y, model.predict_proba(X)[:, 1])},
        feature_names=feature_names,
        data_window={"train_end": params["split"]["train_end"]},
    )
    save_model(model, card, settings.model_path)
    print(f"Saved model -> {settings.model_path}")


if __name__ == "__main__":
    main()
