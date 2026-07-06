"""Logistic-regression baseline.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    Baseline first, always. "We beat the baseline by X" is the only meaningful
    framing for model complexity; a GBDT that ties a logistic regression is a loss.

KEY CONCEPT — the Pipeline IS the leakage defense
    Imputation, scaling and encoding are all fit inside the Pipeline, so calling
    .fit(X_train) structurally guarantees no statistic is learned from valid/test.
    Preprocessing outside a Pipeline is where "fit on the full dataset" bugs live.

INTERVIEW ANGLE
    "What model would you start with?" → a regularized logistic regression in a
    Pipeline: fast, calibrated-ish, interpretable coefficients, and it sets the bar
    that justifies (or kills) anything fancier.

TEST
    tests/test_models.py::test_baseline_pipeline_predicts_probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_baseline(numeric: list[str], categorical: list[str], *, seed: int = 42) -> Pipeline:
    """Unfitted Pipeline: impute+scale numerics, impute+one-hot categoricals, LogReg.

    `class_weight="balanced"` is the baseline-side imbalance handling (the GBDT uses
    scale_pos_weight instead). `handle_unknown="ignore"` keeps serving from crashing
    on a category level the training window never produced. Default seed mirrors
    config/params.yaml `seed`.
    """
    transformers = []
    if numeric:
        numeric_prep = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("num", numeric_prep, numeric))
    if categorical:
        categorical_prep = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_prep, categorical))
    if not transformers:
        raise ValueError("need at least one of `numeric` / `categorical` columns")

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def fit_predict_proba(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_score: pd.DataFrame
) -> np.ndarray:
    """Fit on train only, return P(y=1) for X_score — the honest scoring shape."""
    pipe.fit(X_train, y_train)
    return np.asarray(pipe.predict_proba(X_score)[:, 1])
