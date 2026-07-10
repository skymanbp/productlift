"""Load a trained model and score products. MOSTLY BUILT.

Wraps the registry so the API doesn't know how models are stored. Keeping the
train/serve symmetry (same feature names, same preprocessing baked into the
Pipeline) is the production lesson here: skew between train and serve is a top
cause of silent model failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.models.registry import ModelCard, load_model


class Predictor:
    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model, self.card = self._load(model_path)
        self.threshold = threshold

    @staticmethod
    def _load(path: Path) -> tuple[Any, ModelCard]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No model at {path}. Train one first: `make train`."
            )
        return load_model(Path(path))

    def predict_one(self, features: dict) -> tuple[float, bool]:
        """Return (probability, flagged) for a single product's feature dict.

        Builds a 1-row frame in the model's expected column order, then scores.
        Preprocessing lives inside the Pipeline, so we pass raw-ish features and
        the model handles imputation/encoding/scaling as it did in training.

        None (JSON null, the API's missing-value spelling) becomes NaN before the
        frame is built: a 1-row column holding None infers object dtype, which the
        numeric preprocessing rejects, whereas NaN keeps the column float64 so the
        missing value flows through imputation like it did in training. Batch
        frames never hit this because a column with any real value stays numeric.
        """
        cols = self.card.feature_names or list(features.keys())
        values = {c: features.get(c) for c in cols}
        row = pd.DataFrame([{c: float("nan") if v is None else v for c, v in values.items()}])
        proba = float(self.model.predict_proba(row)[:, 1][0])
        return proba, proba >= self.threshold
