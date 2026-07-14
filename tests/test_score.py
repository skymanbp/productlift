"""Tests for scripts/score.py (batch scoring / intervention ranking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.gbdt import build_gbdt, compute_scale_pos_weight, fit_gbdt
from app.models.registry import ModelCard
from scripts.score import rank_products

TARGET = "is_underperforming"
NUMERIC = ["x1", "x2"]


@pytest.fixture
def scored_setup(modeling_matrix):
    matrix = modeling_matrix.copy()
    matrix.insert(0, "product_id", [f"p{i}" for i in range(len(matrix))])
    X = matrix[NUMERIC]
    y = matrix[TARGET]
    model = build_gbdt(scale_pos_weight=compute_scale_pos_weight(y))
    model = fit_gbdt(model, X, y, X, y)
    card = ModelCard(name="score-fixture-gbdt", feature_names=NUMERIC)
    return matrix, model, card


def test_rank_products_sorted_and_valid(scored_setup):
    matrix, model, card = scored_setup
    ranked = rank_products(matrix, model, card)
    assert len(ranked) == len(matrix)
    assert ranked["risk_score"].between(0.0, 1.0).all()
    # highest risk first, rank column runs 1..n in that order
    assert (ranked["risk_score"].to_numpy() == np.sort(ranked["risk_score"])[::-1]).all()
    assert ranked["rank"].tolist() == list(range(1, len(matrix) + 1))
    assert ranked["product_id"].is_unique


def test_rank_products_scores_match_batch_predict(scored_setup):
    # Scores must be the model's own probabilities, aligned to each product.
    matrix, model, card = scored_setup
    ranked = rank_products(matrix, model, card)
    expected = pd.Series(
        model.predict_proba(matrix[card.feature_names])[:, 1],
        index=matrix["product_id"].to_numpy(),
    )
    aligned = ranked.set_index("product_id")["risk_score"]
    assert np.allclose(aligned.to_numpy(), expected.loc[aligned.index].to_numpy())


def test_rank_products_missing_feature_fails_loud(scored_setup):
    matrix, model, card = scored_setup
    with pytest.raises(ValueError, match="missing model features"):
        rank_products(matrix.drop(columns=["x2"]), model, card)
