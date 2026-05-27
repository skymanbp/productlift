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


def temporal_split(
    df: pd.DataFrame, time_col: str, train_end: str, valid_end: str
) -> Split:
    """Split rows by `time_col`: train ≤ train_end < valid ≤ valid_end < test.

    TODO(lillian):
      - parse train_end / valid_end to timestamps (pd.Timestamp)
      - train  = rows with time_col <= train_end
      - valid  = rows with train_end < time_col <= valid_end
      - test   = rows with time_col >  valid_end
      - return Split(train, valid, test) with index reset
    The test asserts the windows don't overlap and that max(train.time) <
    min(valid.time) — i.e. no temporal leakage across the boundary.
    """
    raise NotImplementedError("Implement temporal_split — see tests/test_splits.py")
