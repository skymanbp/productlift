"""Leakage-safe categorical encoding.  ⛏️  EXERCISE — Milestone 2 (high value).

LEARNING GOAL
    product_category_name has ~70 levels: too many for one-hot, good for target
    encoding. Naive target encoding leaks the label into the feature (a row
    encoded with a mean that includes its own target).

KEY CONCEPT — smoothing + train-only fit (+ why OOF matters)
    encoding(c) = (n_c * mean_c + m * global_mean) / (n_c + m); rare levels
    shrink toward the global mean (Bayesian smoothing), unseen levels get the
    global mean. Fit on train only. For the training fold, out-of-fold encoding
    keeps a row's own label out of its own feature; valid/test use the full-train
    mapping. SmoothedTargetEncoder is the fit/transform mapping; oof_target_encode
    is the OOF version for the training design matrix.

INTERVIEW ANGLE
    High-cardinality categorical -> smoothed target encoding, train-only fit, OOF
    for the training fold, prior for unseen levels. Naming the leakage trap
    unprompted is a strong signal.

TEST
    tests/test_encoders.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoder for one categorical column.

    Implements the sklearn fit/transform contract so it drops into a Pipeline,
    which keeps the fit on train data only.
    """

    def __init__(self, column: str, smoothing: float = 20.0) -> None:
        self.column = column
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SmoothedTargetEncoder:
        """Learn the smoothed per-category target mean from train data.

        NaN categories get no mapping entry, so at transform time they fall back
        to the global mean like any unseen level. X and y are paired positionally,
        not by index, so a sliced X with a re-indexed y cannot mis-align.
        """
        if self.column not in X.columns:
            raise ValueError(f"column {self.column!r} not in X")
        target = np.asarray(y, dtype=float)
        if len(target) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(target)}")

        self.global_mean_ = float(target.mean())
        stats = (
            pd.DataFrame({"category": X[self.column].to_numpy(), "target": target})
            .groupby("category")["target"]
            .agg(["mean", "count"])
        )
        m = self.smoothing
        smoothed = (stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m)
        self.mapping_ = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map X[self.column] to its encoded value; unseen categories get global_mean_.

        Returns shape (n, 1) so it composes in a ColumnTransformer.
        """
        if not hasattr(self, "mapping_"):
            raise ValueError("SmoothedTargetEncoder.transform called before fit")
        encoded = X[self.column].map(self.mapping_).fillna(self.global_mean_)
        return encoded.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Name the single output column so feature names propagate through a
        ColumnTransformer with set_output(transform="pandas") and downstream
        estimators see the same names at fit and predict time."""
        return np.asarray([f"{self.column}_target_encoded"], dtype=object)


def oof_target_encode(
    X: pd.DataFrame,
    y: pd.Series,
    column: str,
    *,
    smoothing: float = 20.0,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold target encoding for the train fold.

    fit-then-transform on train leaks: each row's own label sits in the category
    mean that becomes its feature, so the model overfits the encoded column on the
    train fold. Valid/test stay clean (their labels never entered the mapping),
    which hides the leak in holdout metrics. OOF splits train into K folds and
    encodes each row with an encoder fit on the other folds, so its own label
    cannot reach its own feature (categories absent from those folds fall back to
    the fold's prior, like an unseen level at serving). Deployment still uses the
    full-train mapping; OOF only keeps the training design matrix honest.

    Returns shape (n, 1), matching SmoothedTargetEncoder.transform.
    """
    target = np.asarray(y, dtype=float)
    if len(target) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(target)}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    out = np.empty(len(X), dtype=float)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, encode_idx in folds.split(X):
        enc = SmoothedTargetEncoder(column, smoothing=smoothing)
        enc.fit(X.iloc[fit_idx], pd.Series(target[fit_idx]))
        out[encode_idx] = enc.transform(X.iloc[encode_idx]).ravel()
    return out.reshape(-1, 1)
