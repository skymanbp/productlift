"""Tests for observability/monitoring.py + scripts/monitor.py (drift loop)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observability.drift import population_stability_index
from observability.monitoring import drift_report
from scripts.monitor import breached

RNG = np.random.default_rng(7)


def test_psi_refuses_nan():
    # NaN quantile edges collapse every observation into one bin and read "stable"
    # whatever the drift, so the monitor must fail loud rather than return a number.
    clean = RNG.normal(size=200)
    dirty = np.concatenate([RNG.normal(size=199), [np.nan]])
    with pytest.raises(ValueError, match="NaN"):
        population_stability_index(dirty, clean)
    with pytest.raises(ValueError, match="NaN"):
        population_stability_index(clean, dirty)


def test_drift_report_splits_psi_and_missingness():
    n = 500
    ref = pd.DataFrame({"shifted": RNG.normal(0, 1, n), "gappy": RNG.normal(0, 1, n)})
    cur = pd.DataFrame({"shifted": RNG.normal(2, 1, n), "gappy": RNG.normal(0, 1, n)})
    cur.loc[: n // 2, "gappy"] = np.nan  # same values, missingness jumped

    report = drift_report(ref, cur, ["shifted", "gappy"]).set_index("feature")
    assert report.loc["shifted", "status"] == "significant"
    # gappy's observed values didn't move, so PSI stays calm; the missing-rate
    # channel is what carries the alarm.
    assert report.loc["gappy", "status"] == "stable"
    assert report.loc["gappy", "missing_ref"] == 0.0
    assert report.loc["gappy", "missing_cur"] > 0.4


def test_drift_report_no_data_status():
    ref = pd.DataFrame({"dead": [np.nan] * 50})
    cur = pd.DataFrame({"dead": RNG.normal(size=50)})
    report = drift_report(ref, cur, ["dead"])
    assert report.loc[0, "status"] == "no_data"


def test_breached_levels():
    report = pd.DataFrame({"feature": ["a", "b"], "status": ["stable", "moderate"]})
    assert not breached(report, "significant")
    assert breached(report, "moderate")
    # no_data trips every level: zero observations means a broken pipeline.
    dead = pd.DataFrame({"feature": ["a"], "status": ["no_data"]})
    assert breached(dead, "significant")
    with pytest.raises(ValueError, match="fail_on"):
        breached(report, "stable")
