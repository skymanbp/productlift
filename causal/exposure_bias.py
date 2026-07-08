"""Exposure / position-bias correction.  ⛏️  EXERCISE — Milestone 6 (the signature one).

LEARNING GOAL
    Operationalize the interview doc's killer question: "Is the product bad, or is it
    just badly ranked / under-exposed?" A product low in search results gets few
    clicks regardless of quality — so raw conversion conflates true quality with
    position. This module estimates quality with position held constant.

KEY CONCEPT — position as a confounder; IPW by examination propensity
    Position affects exposure (whether the user even sees the item), which affects
    conversion. If we don't adjust, we punish good products that happen to rank low —
    a feedback loop that buries them further. The standard fix borrows from
    counterfactual learning-to-rank: weight each observation by the inverse of its
    position's examination probability (the propensity of being seen), so clicks at
    low-exposure positions count for more.

INTERVIEW ANGLE
    This is exactly the "exposure bias / ranking position causes underperformance"
    point from the doc. Being able to NAME the confound and propose IPW-by-position
    (or a randomized position experiment to estimate the examination curve) is a
    senior-level answer.

TEST
    tests/test_exposure_bias.py  (synthetic: equal-quality items at different
    positions should get equal corrected scores, unequal raw scores)
"""

from __future__ import annotations

import pandas as pd


def position_adjusted_rate(
    df: pd.DataFrame,
    *,
    item_col: str = "product_id",
    position_col: str = "position",
    click_col: str = "click",
    examination_by_position: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Return a per-item position-corrected conversion rate.

    `examination_by_position` maps position → P(examined); if None, assume the
    classic 1/position examination decay (documented default; in practice you
    estimate this curve from position-randomized traffic).

    Estimator: corrected_rate = mean(click / examination) per item — the
    inverse-propensity-scoring estimate of P(click | examined). It is unbiased
    because E[click] = examination × true_ctr, so dividing each observation by
    its examination probability recovers true_ctr. NOTE: the exercise TODO
    originally suggested sum(click·w)/sum(w), but with one fixed position per
    item that self-normalization cancels the constant weight and degenerates to
    the naive rate — it cannot pass the test and is not the IPS estimator.
    Examination probabilities are floored at `min_examination` before inverting
    (a 0.1%-examined position would otherwise carry weight 1000 and let one
    noisy click dominate the estimate).
    """
    required = {item_col, position_col, click_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")

    min_examination = 0.01
    if examination_by_position is None:
        examination = 1.0 / df[position_col].astype(float)
    else:
        unknown = set(df[position_col].unique()) - set(examination_by_position)
        if unknown:
            raise ValueError(f"positions without an examination probability: {sorted(unknown)}")
        examination = df[position_col].map(examination_by_position).astype(float)
    if (examination <= 0).any() or (examination > 1).any():
        raise ValueError("examination probabilities must lie in (0, 1]")

    work = df[[item_col, click_col]].copy()
    work["_ips"] = work[click_col] / examination.clip(lower=min_examination)

    out = work.groupby(item_col, as_index=False).agg(
        naive_rate=(click_col, "mean"),
        corrected_rate=("_ips", "mean"),
        n=(click_col, "size"),
    )
    return out
