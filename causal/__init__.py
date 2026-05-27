"""Causal inference: separating correlation from causation.

The interview doc's sharpest question lives here: "How do you know underperformance
is caused by the product itself and not by ranking position or lack of exposure?"
Answering it needs propensity/IPW (adjust for confounding), uplift (who to treat),
and exposure-bias correction (the position confound specifically).

These are the tools that let you say "we ESTIMATE the effect of X" instead of "X is
correlated with Y" — the line between an analyst and an applied scientist.
"""
