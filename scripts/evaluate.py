"""Evaluate the registered model on a holdout and write a report. MOSTLY BUILT.

Runs once Milestone 4 (metrics + harness) is implemented.

    python -m scripts.evaluate
"""

from __future__ import annotations

import pandas as pd

from app.config import get_settings
from app.models.registry import load_model
from evaluation.offline_eval import evaluate_predictions, save_run
from evaluation.reports import format_report

TARGET = "is_underperforming"


def main() -> None:
    settings = get_settings()
    matrix = pd.read_parquet(settings.processed_dir / "modeling_matrix.parquet")
    model, card = load_model(settings.model_path)

    y = matrix[TARGET]
    X = matrix.drop(columns=[TARGET])
    y_prob = model.predict_proba(X)[:, 1]

    segments = X["product_category_name"] if "product_category_name" in X else None
    report = evaluate_predictions(y.to_numpy(), y_prob, segments=segments)
    path = save_run(report)
    print(format_report(report))
    print(f"\nSaved run -> {path}")


if __name__ == "__main__":
    main()
