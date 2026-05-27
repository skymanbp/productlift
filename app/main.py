"""FastAPI scoring service. FULLY BUILT (light).

Boots and serves /health even with no model trained yet; /predict comes alive once
you've run `make train`. Interactive docs at http://localhost:8000/docs.

Serving is intentionally the *lighter* part of this repo — the modeling, stats, and
causal work is the focus. But shipping a model behind an API, in a container, is the
"production-ready" signal employers look for, so it's here end to end.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.schemas import HealthResponse, PredictRequest, PredictResponse

log = structlog.get_logger()
app = FastAPI(title="ProductLift", version="0.1.0",
              description="Scores e-commerce products for underperformance risk.")

# Lazy singleton so the app boots even before a model exists.
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from app.serving.predictor import Predictor

        _predictor = Predictor(get_settings().model_path)
    return _predictor


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        p = _get_predictor()
        return HealthResponse(status="ok", model_loaded=True, model_name=p.card.name)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        p = _get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    proba, flagged = p.predict_one(req.features)
    return PredictResponse(
        product_id=req.product_id,
        underperformance_probability=proba,
        flagged=flagged,
        model_name=p.card.name,
        model_version=p.card.created_at,
    )
