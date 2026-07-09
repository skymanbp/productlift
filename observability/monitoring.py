"""Monitoring helpers. FULLY BUILT (light).

Run PSI across many features and flag the ones that breached a threshold. Meant
to be called from a scheduled job (scripts/monitor.py); in a real deployment it
would feed a dashboard or alerting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from observability.drift import population_stability_index

PSI_THRESHOLDS = {"stable": 0.1, "moderate": 0.25}


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Per feature: PSI over observed values, plus the missing rate in each window.

    PSI itself refuses NaN (its quantile edges would degenerate), so missingness
    is handled here as a second signal instead of folded in: PSI over the values
    that exist, and the missing rate per window. A missingness shift is drift too
    (an upstream join breaking shows up as missing_cur rising, not as a PSI move).
    A feature with no observed values in either window gets status "no_data".
    """
    rows = []
    for f in features:
        ref = reference[f].to_numpy(dtype=float)
        cur = current[f].to_numpy(dtype=float)
        ref_obs = ref[~np.isnan(ref)]
        cur_obs = cur[~np.isnan(cur)]
        if len(ref_obs) == 0 or len(cur_obs) == 0:
            psi, status = float("nan"), "no_data"
        else:
            psi = population_stability_index(ref_obs, cur_obs)
            if psi < PSI_THRESHOLDS["stable"]:
                status = "stable"
            elif psi < PSI_THRESHOLDS["moderate"]:
                status = "moderate"
            else:
                status = "significant"
        rows.append(
            {
                "feature": f,
                "psi": psi,
                "status": status,
                "missing_ref": float(np.isnan(ref).mean()),
                "missing_cur": float(np.isnan(cur).mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False, na_position="first")
        .reset_index(drop=True)
    )
