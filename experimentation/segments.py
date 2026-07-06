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
from statsmodels.stats.multitest import multipletests

from experimentation.analyze import two_proportion_test


def segment_effects(
    df: pd.DataFrame, segment_col: str, group_col: str, outcome_col: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Per segment level, estimate the treatment effect and test it; return a frame
    with raw and multiplicity-corrected significance.

    `df` has one row per unit with a segment, a group (control/treatment), and a
    binary outcome. Correction is Benjamini-Hochberg (controls the false discovery
    rate): with many exploratory segments, FDR "of the winners we call, few are
    flukes" matches how segment scans are used better than Bonferroni's "no false
    positive anywhere", which sacrifices most of the power. A segment missing one
    of the two arms is an experiment-design bug, not a statistics question — it
    raises rather than being silently skipped.
    """
    required = {segment_col, group_col, outcome_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")
    groups = set(df[group_col].unique())
    if groups != {"control", "treatment"}:
        raise ValueError(
            f"{group_col} must contain exactly control/treatment, got {sorted(groups)}"
        )

    rows = []
    for level, sub in df.groupby(segment_col):
        control = sub.loc[sub[group_col] == "control", outcome_col]
        treatment = sub.loc[sub[group_col] == "treatment", outcome_col]
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError(f"segment {level!r} lacks one of the arms — check the assignment")
        res = two_proportion_test(
            int(control.sum()), len(control), int(treatment.sum()), len(treatment), alpha=alpha
        )
        rows.append(
            {
                "segment": level,
                "n": len(sub),
                "lift": res.relative_lift,
                "p_value": res.p_value,
            }
        )

    out = pd.DataFrame(rows)
    rejected, p_adjusted, _, _ = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")
    out["p_adjusted"] = p_adjusted
    out["significant_adj"] = rejected
    return out
