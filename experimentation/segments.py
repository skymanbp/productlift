"""Segment / heterogeneous-effect analysis.  ⛏️  EXERCISE — Milestone 5.

LEARNING GOAL
    An average effect can hide that the treatment helps one group and hurts another.
    Segmenting reveals heterogeneity — but slicing many ways inflates false
    positives, so you must correct for multiple testing. Both halves are interview
    favorites ("what if only one segment benefits?", "how do you handle multiple
    comparisons?").

KEY CONCEPT — multiple testing correction
    Test the effect within each segment, collect the p-values, then control the
    family-wise error (Bonferroni: compare each p to α/m) or the false discovery rate
    (Benjamini-Hochberg, less conservative). Reporting 12 segment tests at α=0.05
    without correction means ~46% chance of a false "winner" by luck alone.

INTERVIEW ANGLE
    "What if results conflict across segments?" → report the heterogeneity honestly,
    correct for multiplicity, and distinguish pre-registered segments (confirmatory)
    from exploratory ones (hypothesis-generating, not conclusions).

TEST
    tests/test_segments.py
"""

from __future__ import annotations

import pandas as pd


def segment_effects(
    df: pd.DataFrame, segment_col: str, group_col: str, outcome_col: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Per segment level, estimate the treatment effect and test it; return a frame
    with raw and multiplicity-corrected significance.

    `df` has one row per unit with a segment, a group (control/treatment), and a
    binary outcome.

    TODO(lillian):
      - for each segment level: run two_proportion_test (reuse analyze.py) on
        control vs treatment within that segment → effect + p-value
      - collect p-values; apply Benjamini-Hochberg (or Bonferroni) across segments
      - return a DataFrame: [segment, n, lift, p_value, p_adjusted, significant_adj]
    The test checks that adding more segments makes the correction stricter.
    """
    raise NotImplementedError("Implement segment_effects — see tests/test_segments.py")
