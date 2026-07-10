"""Regression tests for the serving boundary (app/serving/predictor.py).

Found during the M8 smoke test: a /predict request carrying a JSON-null numeric
feature (e.g. trend_30d for a product with no prior-window base) built a 1-row
frame whose column dtype was object. Batch scoring never hits this, since a
column with any real value alongside NaN stays float, so it is exactly the
single-row train/serve skew the predictor docstring warns about.

The fixture head must be LightGBM, matching the production artifact: sklearn
preprocessing silently coerces an object column of [None] to float, so a
baseline-pipeline fixture would pass even without the fix. LightGBM rejects
object dtype outright ("Fields with bad pandas dtypes"), which is what made the
live API 500 and what pins the regression here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.gbdt import build_gbdt, compute_scale_pos_weight, fit_gbdt
from app.models.registry import ModelCard, save_model
from app.serving.predictor import Predictor

TARGET = "is_underperforming"
NUMERIC = ["x1", "x2"]


@pytest.fixture
def served_model_path(modeling_matrix, tmp_path):
    """A fitted GBDT saved through the registry, loaded the way serving loads it."""
    X = modeling_matrix[NUMERIC]
    y = modeling_matrix[TARGET]
    model = build_gbdt(scale_pos_weight=compute_scale_pos_weight(y))
    model = fit_gbdt(model, X, y, X, y)
    card = ModelCard(name="serving-fixture-gbdt", feature_names=NUMERIC)
    return save_model(model, card, tmp_path / "current.joblib")


def test_predict_one_imputes_missing_numeric_feature(served_model_path):
    # None (JSON null) for a numeric feature must reach the model as float NaN
    # (LightGBM handles NaN natively), not as an object-dtype column it rejects.
    predictor = Predictor(served_model_path)
    proba, flagged = predictor.predict_one({"x1": None, "x2": 0.3})
    assert 0.0 <= proba <= 1.0
    assert isinstance(flagged, bool)


def test_predict_one_matches_batch_scoring(served_model_path):
    # Train/serve consistency: the 1-row serving path must reproduce the score
    # the same model gives the same product in a batch frame.
    predictor = Predictor(served_model_path)
    features = {"x1": 0.5, "x2": -1.2}
    proba, _ = predictor.predict_one(features)
    batch = float(predictor.model.predict_proba(pd.DataFrame([features]))[:, 1][0])
    assert proba == pytest.approx(batch)
