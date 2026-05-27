"""API data contracts for the scoring service. FULLY BUILT.

The serving boundary takes a product's features and returns a calibrated
probability of underperformance plus the decision at the operating threshold.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Features for one product at scoring time. Mirror the training feature names.
    Kept permissive (extra fields ignored) so the schema can evolve; in a stricter
    setup you'd validate the full feature contract here."""

    model_config = {"extra": "allow"}

    product_id: str
    features: dict[str, float | int | str | None] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    product_id: str
    underperformance_probability: float
    flagged: bool          # probability >= operating threshold
    model_name: str
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
