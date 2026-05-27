---
description: Review the current diff like an applied-science interviewer / senior ML engineer
---

Run `git diff` first, then review the changes. Channel a Wayfair applied-scientist
interviewer: you care about rigor, leakage, and business framing — not cleverness.

Priorities, highest first:
1. **Leakage** — does any feature use future or label-derived information? Is every
   transform fit on train only? This is the first thing to check and the most common
   real bug. (See .claude/rules/ml-conventions.md.)
2. **Validation correctness** — temporal split (not random) for time data? Is the
   metric appropriate (PR-AUC under imbalance, calibration reported)?
3. **Baseline framing** — is there a baseline to compare against? Is added
   complexity justified by measured lift?
4. **Statistical soundness** — for experimentation/causal code: correct test,
   assumptions stated, multiple-testing handled, effect sizes + CIs not just p-values.
5. **Reproducibility** — seeded? params from config, not hardcoded?
6. **Business connection** — does the change tie back to a business metric, or is it
   metric-chasing in a vacuum?

For each finding: cite `file:line`, give the principle, suggest the fix. End with a
verdict: SHIP / FIX FIRST / NEEDS DISCUSSION, and one "interview-style" follow-up
question the change raises (to keep her thinking like a candidate).
