"""Train/validation/test splitting.  ⛏️  EXERCISE — Milestone 1.

LEARNING GOAL
    Split correctly. For time-ordered data this is the difference between a model
    that works and a number that lies. The single most common interview-killer is
    random-splitting time series.

KEY CONCEPT — temporal splits
    Train on the past, validate/test on the future, exactly as the model will be
    used in production. A random split lets the model "see the future" (e.g. a
    seasonal spike present in both train and test), inflating offline metrics that
    then collapse in production.

INTERVIEW ANGLE
    "How would you validate this?" → "Temporal split, because the data is a time
    series and we predict forward. I'd train on data up to T1, validate on T1–T2,
    test on >T2, and make sure no feature uses information after the prediction
    timestamp." That sentence alone signals seniority.

TEST
    tests/test_splits.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, time_col: str, train_end: str, valid_end: str) -> Split:
    """Split rows by `time_col`: train ≤ train_end < valid ≤ valid_end < test.

    Boundary rows go to the EARLIER window (train gets `train_end` itself) so a
    row can never appear on both sides — the no-leakage guarantee the tests
    assert. The three masks are mutually exclusive and exhaustive, which is only
    true if `time_col` has no NaT (NaT compares False everywhere and would be
    silently dropped from all three windows), so we fail loud on that.
    """
    t1 = pd.Timestamp(train_end)
    t2 = pd.Timestamp(valid_end)
    if not t1 < t2:
        raise ValueError(f"train_end ({t1}) must be strictly before valid_end ({t2})")

    t = df[time_col]
    if t.isna().any():
        raise ValueError(
            f"{time_col} contains {int(t.isna().sum())} NaT rows; they would be "
            "silently dropped from every window — clean them upstream first"
        )

    train = df[t <= t1].reset_index(drop=True)
    valid = df[(t > t1) & (t <= t2)].reset_index(drop=True)
    test = df[t > t2].reset_index(drop=True)
    return Split(train=train, valid=valid, test=test)
