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

import numpy as np
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

    `examination_by_position` maps position → P(examined); if None, assume a simple
    decay (you choose and document it, e.g. 1/position).

    TODO(lillian):
      - per row, weight = 1 / examination(position)  (clip to avoid blow-ups)
      - per item, corrected_rate = sum(click * weight) / sum(weight)
      - also return the naive (unweighted) rate so you can SHOW the difference
      - return [item, naive_rate, corrected_rate, n].
    The test puts equal-quality items at good vs bad positions and checks the
    corrected rates converge while the naive ones don't.
    """
    raise NotImplementedError("Implement position_adjusted_rate — see tests/test_exposure_bias.py")
