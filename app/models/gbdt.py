"""LightGBM classifier with imbalance handling and early stopping.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    The production default on tabular data — but justified by measured lift over the
    baseline, tuned for small data, and stopped early on a real validation fold.

KEY CONCEPT — scale_pos_weight + early stopping on the metric you care about
    scale_pos_weight = n_neg/n_pos upweights the minority class in the loss.
    Early stopping monitors average_precision (= PR-AUC) on the VALID fold, so the
    stopping rule optimizes the same metric we report — stopping on AUC while
    reporting PR-AUC quietly optimizes the wrong thing under imbalance.

INTERVIEW ANGLE
    "How do you handle class imbalance?" → reweight the loss (scale_pos_weight /
    class_weight), never oversample before the split; and evaluate with PR-AUC plus
    calibration, not accuracy.

TEST
    tests/test_models.py::{test_compute_scale_pos_weight, test_gbdt_fits_and_predicts}
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def compute_scale_pos_weight(y: pd.Series) -> float:
    """n_negative / n_positive on the TRAIN fold (computing it on all data would
    leak the holdout's class balance into training)."""
    arr = np.asarray(y)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    if n_pos == 0:
        raise ValueError("y has no positive samples — scale_pos_weight is undefined")
    return n_neg / n_pos


def build_gbdt(
    scale_pos_weight: float | None = None,
    *,
    seed: int = 42,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 15,
) -> LGBMClassifier:
    """Unfitted LGBMClassifier. Defaults mirror config/params.yaml `model.gbdt`
    (small num_leaves because the matrix is hundreds of rows, not millions);
    n_estimators is a ceiling — early stopping in fit_gbdt picks the real count."""
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
    )


def fit_gbdt(
    model: LGBMClassifier,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series,
    X_valid: pd.DataFrame | np.ndarray,
    y_valid: pd.Series,
    *,
    early_stopping_rounds: int = 100,
) -> LGBMClassifier:
    """Fit with early stopping on the valid fold, monitoring average precision.

    Accepts ndarray as well as DataFrame because the training script feeds it the
    output of a fitted ColumnTransformer.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model
