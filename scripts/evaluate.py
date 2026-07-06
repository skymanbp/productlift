"""Evaluate the registered model on the untouched TEST fold and write a report.

The scaffold intent ("evaluate on a holdout") made honest: we reproduce the same
seeded product split scripts/train.py used, score ONLY the test fold, and slice
the report by price band — the matrix's 177 test rows spread over ~40 categories
would leave every category segment below any credible sample size.

    python -m scripts.evaluate
"""

from __future__ import annotations

import pandas as pd

from app.config import get_params, get_settings
from app.models.registry import load_model
from evaluation.offline_eval import evaluate_predictions, save_run
from evaluation.reports import format_report
from scripts.train import _seeded_product_split

TARGET = "is_underperforming"


def main() -> None:
    settings = get_settings()
    params = get_params()
    matrix = pd.read_parquet(settings.processed_dir / "modeling_matrix.parquet")
    model, card = load_model(settings.model_path)

    # Same seed + fractions as training -> identical fold membership; test fold
    # is the only slice the model has never influenced in any way.
    holdout = params["model"]["holdout"]
    _, _, _, idx_te = _seeded_product_split(
        len(matrix),
        test_fraction=holdout["test_fraction"],
        valid_fraction=holdout["valid_fraction"],
        calib_fraction=holdout["calib_fraction"],
        seed=int(params["seed"]),
    )
    test = matrix.iloc[idx_te]

    # card.feature_names IS the serving contract — slice by it, in its order.
    X_te = test[card.feature_names]
    y_te = test[TARGET]
    y_prob = model.predict_proba(X_te)[:, 1]

    price_band = pd.qcut(
        X_te["avg_unit_price"], q=3, labels=["price_low", "price_mid", "price_high"]
    )
    report = evaluate_predictions(y_te.to_numpy(), y_prob, segments=price_band)
    path = save_run(report)
    print(format_report(report))
    print(f"\nSaved run -> {path}")


if __name__ == "__main__":
    main()
