"""Analyze a completed A/B test.  ⛏️  EXERCISE — Milestone 5.

LEARNING GOAL
    Given control/treatment outcomes, produce the full answer an interviewer wants:
    not just a p-value, but the effect size, a confidence interval, and a guardrail
    check. "p < 0.05" alone is a junior answer; "+3.2% relative lift, 95% CI
    [1.1%, 5.3%], p=0.004, no guardrail regression" is a senior one.

KEY CONCEPT — two-proportion z-test + CI on the difference
    For conversion rates: pooled-proportion z-test for the p-value; for the CI use
    the UNpooled standard error of (p_t - p_c):
        SE = sqrt( p_c(1-p_c)/n_c + p_t(1-p_t)/n_t )
        CI = (p_t - p_c) ± z_{1-α/2} * SE
    Report absolute diff, relative lift, the CI, and the p-value.

INTERVIEW ANGLE
    Always pair the primary metric with GUARDRAILS (the doc: latency, return rate,
    satisfaction). "Conversion improved but returns also rose" is a real possible
    outcome — surface it, don't bury it.

TEST
    tests/test_ab_analyze.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class ABResult:
    control_rate: float
    treatment_rate: float
    absolute_diff: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


def two_proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ABResult:
    """Run a two-proportion z-test and return effect size + CI + significance.

    Pooled SE for the TEST (under H0 the two arms share one rate — pooling is the
    null's own assumption); UNpooled SE for the CI (the interval describes the
    actual two-rate world, not the null). Degenerate data (both arms all-0 or
    all-1) has zero pooled SE: we report p=1.0 — no evidence of a difference —
    rather than dividing by zero. relative_lift is NaN when the control rate is 0
    (a lift relative to nothing is undefined).
    """
    for name, conv, n in (
        ("control", control_conversions, control_n),
        ("treatment", treatment_conversions, treatment_n),
    ):
        if n <= 0:
            raise ValueError(f"{name}_n must be positive, got {n}")
        if not 0 <= conv <= n:
            raise ValueError(f"{name}_conversions={conv} outside [0, n={n}]")

    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    diff = p_t - p_c
    relative_lift = diff / p_c if p_c > 0 else float("nan")

    pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    if se_pooled == 0.0:
        p_value = 1.0
    else:
        z = diff / se_pooled
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))

    se_unpooled = math.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
    half_width = float(norm.ppf(1.0 - alpha / 2.0)) * se_unpooled

    return ABResult(
        control_rate=p_c,
        treatment_rate=p_t,
        absolute_diff=diff,
        relative_lift=relative_lift,
        ci_low=diff - half_width,
        ci_high=diff + half_width,
        p_value=p_value,
        significant=bool(p_value < alpha),
    )


# Direction convention: True = a HIGHER rate is WORSE for the business (return
# rate going up is bad); False = higher is better (satisfaction going up is good).
# Explicit registry instead of name heuristics — a guardrail whose direction we
# never declared should fail loudly, not be guessed.
HIGHER_IS_WORSE: dict[str, bool] = {
    "return_rate": True,
    "complaint_rate": True,
    "latency": True,
    "cancellation_rate": True,
    "satisfaction": False,
    "review_score": False,
    "repeat_purchase_rate": False,
}


def check_guardrails(
    results: dict[str, ABResult],
    guardrail_metrics: list[str],
    *,
    higher_is_worse: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Return {guardrail: passed?} — a guardrail FAILS if treatment SIGNIFICANTLY
    moved it in the bad direction.

    "Significantly" matters: a noisy wiggle in the bad direction is not a failed
    guardrail, and treating it as one would veto every experiment. Directions come
    from HIGHER_IS_WORSE, overridable per call; an unregistered metric raises —
    guessing which way "worse" points is how return-rate regressions ship.
    """
    directions = {**HIGHER_IS_WORSE, **(higher_is_worse or {})}
    verdict: dict[str, bool] = {}
    for metric in guardrail_metrics:
        if metric not in results:
            raise KeyError(f"no ABResult provided for guardrail {metric!r}")
        if metric not in directions:
            raise ValueError(
                f"direction for guardrail {metric!r} not declared — add it to "
                "HIGHER_IS_WORSE or pass higher_is_worse={...}"
            )
        r = results[metric]
        moved_worse = r.absolute_diff > 0 if directions[metric] else r.absolute_diff < 0
        verdict[metric] = not (r.significant and moved_worse)
    return verdict
