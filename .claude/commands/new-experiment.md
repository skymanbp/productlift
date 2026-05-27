---
description: Scaffold and run a clean modeling experiment with before/after metrics
argument-hint: "[what you're changing, e.g. 'add rolling 30d ATC features']"
---

We're running a modeling experiment: $ARGUMENTS

Treat it like a scientist, not a tinkerer:

1. **State the hypothesis** in one line: "Adding X will improve <primary_metric>
   because <reason>." Write it down before touching code.
2. **Record the baseline.** Run `make evaluate` and capture the current metrics from
   the latest eval_history entry. This is your control.
3. **Make the ONE change.** Change one thing so the result is attributable. If you
   need to change several, that's several experiments.
4. **Re-run** `make features` (if features changed) and `make train && make evaluate`.
5. **Compare** the new metrics to the baseline on the primary metric AND the guardrails
   (calibration, by-segment). Did it actually help, or just move one number?
6. **Decide & log**: keep or revert, and write one line in the experiment log
   (docs or eval_history) on what you learned. A negative result is still a result.

Report the before/after table and your decision.
