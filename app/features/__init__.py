"""Feature engineering.

The doc weights this heavily, and so does the interview. Every function here takes
the FEATURE window (events strictly before the as-of date) and returns a
product-level frame keyed by product_id. Keeping every feature builder pure and
product-keyed makes them composable (features.build joins them) and testable.

The golden rule lives in .claude/rules/ml-conventions.md: a feature may only use
information knowable before the prediction time. If you ever compute a feature from
the outcome window, that's leakage.
"""
