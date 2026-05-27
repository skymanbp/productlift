---
name: eval-report
description: Produce an interview-grade evaluation report for the current model — offline metrics, calibration, by-segment breakdown, and the business framing. Use after training to summarize "is this model actually good?"
---

# Evaluation report

Generate the kind of model write-up an applied scientist presents to stakeholders.

## Steps
1. Run `make evaluate` (or evaluation.offline_eval) on the holdout set.
2. Report the **primary metric** (PR-AUC) with the baseline alongside — lift over
   baseline is the headline, not the absolute number.
3. Report **calibration** (Brier, ECE) and show a reliability summary. A model used
   for ranking/decisions must be calibrated, not just discriminative.
4. **Segment the metrics**: by product category, by price band, by product tenure
   (cold vs established). A model that's great overall but fails on cold products
   may be useless for the actual business goal.
5. **Confusion at the operating point**: at the chosen threshold / top-K%, what's
   precision/recall? Translate to business terms ("we'd flag N products, ~P% truly
   underperforming").
6. **Threats to validity**: leakage checked? temporal split? distribution shift
   between train and test windows?

## Output
A concise markdown report: headline lift, metrics table, segment table, the
operating-point business translation, and a short "what I'd do next" — framed as
what you'd actually say in an interview or a model review.
