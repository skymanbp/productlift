"""Render evaluation reports as markdown. FULLY BUILT.

Turns the dict from offline_eval into the kind of table you'd paste into a model
review or talk through in an interview. Communication is part of the job — a great
result nobody can read is a wasted result.
"""

from __future__ import annotations


def format_report(report: dict, baseline: dict | None = None) -> str:
    lines: list[str] = ["# Model evaluation", "", f"n = {report.get('n', '?')}", ""]
    overall = report.get("overall", {})
    lines.append("## Overall")
    lines.append("| metric | value | baseline | lift |")
    lines.append("|---|---|---|---|")
    base_overall = (baseline or {}).get("overall", {})
    for name, val in overall.items():
        b = base_overall.get(name)
        lift = "" if b is None else f"{val - b:+.4f}"
        lines.append(f"| {name} | {val:.4f} | {('' if b is None else f'{b:.4f}')} | {lift} |")

    by_seg = report.get("by_segment")
    if by_seg:
        lines += ["", "## By segment"]
        any_level = next(iter(by_seg.values()))
        metric_names = list(any_level.keys())
        lines.append("| segment | " + " | ".join(metric_names) + " |")
        lines.append("|" + "---|" * (len(metric_names) + 1))
        for level, vals in by_seg.items():
            cells = " | ".join(
                f"{vals[m]:.4f}" if vals.get(m) is not None else "—" for m in metric_names
            )
            lines.append(f"| {level} | {cells} |")
    return "\n".join(lines)
