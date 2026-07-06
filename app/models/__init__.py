"""Models: baseline, GBDT, calibration, registry.  ⛏️  Milestone 3 (built from scratch).

Kept import-light on purpose — submodules pull in sklearn/lightgbm, so importers
load only what they use (the serving path needs registry, not lightgbm).
"""
