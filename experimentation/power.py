"""Power analysis & experiment design.  ⛏️  EXERCISE — Milestone 5.

LEARNING GOAL
    Decide HOW BIG an experiment must be BEFORE running it. Running an underpowered
    test is the most common applied-science mistake: you "find nothing" not because
    there's no effect but because you couldn't have detected it. Power analysis is
    the antidote and a guaranteed interview topic.

KEY CONCEPT — the four-way tradeoff
    Sample size n, significance α, power (1-β), and minimum detectable effect (MDE)
    are locked together — fix any three, solve for the fourth. For a two-proportion
    test (e.g. conversion rate), the normal-approximation sample size per arm is:

        n ≈ (z_{1-α/2} + z_{1-β})^2 * [p1(1-p1) + p2(1-p2)] / (p2 - p1)^2

    where p1 is baseline and p2 = p1 * (1 + mde_relative).

INTERVIEW ANGLE
    "How long should the experiment run?" → translate required n into days using
    traffic per day, and mention you'd also run at least one full weekly cycle to
    avoid day-of-week and novelty effects.

TEST
    tests/test_power.py  (checks against statsmodels / a hand value)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class DesignResult:
    n_per_arm: int
    baseline_rate: float
    treatment_rate: float
    alpha: float
    power: float


def required_sample_size(
    baseline_rate: float, mde_relative: float, alpha: float = 0.05, power: float = 0.80
) -> DesignResult:
    """Return the required sample size PER ARM to detect a relative lift `mde_relative`.

    Normal-approximation two-proportion formula (KEY CONCEPT above). Defaults for
    alpha/power mirror config/params.yaml `experiment`. The four quantities trade
    off against each other: halving the MDE roughly quadruples n (it enters
    squared in the denominator) — that is why "detect any effect at all" is not a
    designable experiment.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if mde_relative == 0.0:
        raise ValueError("mde_relative = 0 means no effect to detect — n would be infinite")
    p1 = baseline_rate
    p2 = p1 * (1.0 + mde_relative)
    if not 0.0 < p2 < 1.0:
        raise ValueError(f"treatment rate p1*(1+mde) = {p2} outside (0, 1)")

    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    n = math.ceil((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)
    return DesignResult(
        n_per_arm=int(n), baseline_rate=p1, treatment_rate=p2, alpha=alpha, power=power
    )


def days_to_run(n_per_arm: int, daily_traffic_per_arm: int) -> int:
    """Days needed to collect n_per_arm, but never fewer than 7.

    The floor is a design judgment, not math: an experiment that ends mid-week
    confounds the effect with day-of-week patterns, and anything shorter is also
    dominated by novelty effects. Even with traffic to finish in an hour, run the
    full weekly cycle.
    """
    if daily_traffic_per_arm <= 0:
        raise ValueError(f"daily_traffic_per_arm must be positive, got {daily_traffic_per_arm}")
    return max(7, math.ceil(n_per_arm / daily_traffic_per_arm))
