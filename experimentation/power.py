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

from dataclasses import dataclass


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

    TODO(lillian): implement the formula in KEY CONCEPT.
      - p1 = baseline_rate; p2 = p1 * (1 + mde_relative)
      - z_alpha = norm.ppf(1 - alpha/2); z_beta = norm.ppf(power)   (scipy.stats.norm)
      - n = ceil( (z_alpha + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2-p1)^2 )
      - return DesignResult(n_per_arm=n, ...).
    Cross-check your number against statsmodels
    (NormalIndPower / proportion_effectsize) in the test.
    """
    raise NotImplementedError("Implement required_sample_size — see tests/test_power.py")


def days_to_run(n_per_arm: int, daily_traffic_per_arm: int) -> int:
    """Translate required sample size into experiment duration in days.

    TODO(lillian): ceil(n_per_arm / daily_traffic_per_arm), but return at least 7 so
    you always cover a full weekly cycle. State that rule in the docstring of your
    impl — it's a judgment call interviewers want to hear you make.
    """
    raise NotImplementedError
