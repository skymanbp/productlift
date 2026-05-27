"""Monitoring helpers. FULLY BUILT (light).

A thin façade: run PSI across many features and flag the ones that breached a
threshold. In a real deployment this would feed a dashboard / alerting; here it's a
function you can call from a scheduled job.
"""

from __future__ import annotations

import pandas as pd

from observability.drift import population_stability_index

PSI_THRESHOLDS = {"stable": 0.1, "moderate": 0.25}


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per feature, compute PSI(reference, current) and a status label."""
    rows = []
    for f in features:
        psi = population_stability_index(reference[f].to_numpy(), current[f].to_numpy())
        if psi < PSI_THRESHOLDS["stable"]:
            status = "stable"
        elif psi < PSI_THRESHOLDS["moderate"]:
            status = "moderate"
        else:
            status = "significant"
        rows.append({"feature": f, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
