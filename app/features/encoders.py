"""Leakage-safe categorical encoding.  ⛏️  EXERCISE — Milestone 2 (high value).

LEARNING GOAL
    Olist's `product_category_name` has ~70 levels — too many for one-hot, perfect
    for target encoding. But naive target encoding is THE classic leakage bug: if you
    encode a row using the target mean computed over all rows (including itself), you
    leak the label into the feature. You'll build a target encoder that does it
    correctly: smoothed, and fit on training data only.

KEY CONCEPT — smoothing + train-only fit (+ why OOF matters)
    encoding(category) = (n_c * mean_c + m * global_mean) / (n_c + m)
      where n_c = count in that category (train), mean_c = its target mean (train),
      m = smoothing strength. Rare categories shrink toward the global mean — this
      is Bayesian smoothing, and it's what makes the encoder robust to cold
      categories. Unseen categories at transform time get the global mean.
    For training the downstream model, out-of-fold (OOF) encoding avoids leaking the
    train rows into themselves; for valid/test you apply the full-train mapping. We
    implement the fit/transform (train-only) version here; OOF is a documented
    stretch goal in the test file.

INTERVIEW ANGLE
    "How would you encode a high-cardinality categorical?" → smoothed target encoding,
    fit on train only, OOF for the training fold, fallback to the prior for unseen
    levels. Naming the leakage trap unprompted is a strong signal.

TEST
    tests/test_encoders.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """sklearn-compatible smoothed target encoder for a single categorical column.

    Implements the scikit-learn fit/transform contract so it drops straight into a
    Pipeline (which is how we structurally prevent leakage — fit runs on train only).
    """

    def __init__(self, column: str, smoothing: float = 20.0) -> None:
        self.column = column
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmoothedTargetEncoder":
        """Learn the smoothed per-category target mean from TRAIN data only.

        TODO(lillian):
          - self.global_mean_ = y.mean()
          - for each category in X[self.column]: n_c and mean_c of y
          - self.mapping_ = smoothed encoding per category (formula in KEY CONCEPT)
          store enough state to transform unseen data. Return self.
        """
        raise NotImplementedError("Implement fit — see tests/test_encoders.py")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map X[self.column] to its encoded value; unseen categories → global_mean_.

        TODO(lillian): return a 2D array (n, 1) so it composes in a ColumnTransformer.
        The test asserts: (1) a frequent category maps near its raw mean, (2) a rare
        category is pulled toward the global mean, (3) an unseen category gets exactly
        the global mean.
        """
        raise NotImplementedError("Implement transform — see tests/test_encoders.py")
