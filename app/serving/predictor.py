"""Load a trained model and score products. MOSTLY BUILT.

Wraps the registry so the API doesn't know how models are stored. The training/
serving symmetry — same feature names, same preprocessing baked into the Pipeline —
is the production lesson here: skew between train and serve is a top cause of silent
model failures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.models.registry import ModelCard, load_model


class Predictor:
    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model, self.card = self._load(model_path)
        self.threshold = threshold

    @staticmethod
    def _load(path: Path) -> tuple[object, ModelCard]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No model at {path}. Train one first: `make train`."
            )
        return load_model(Path(path))

    def predict_one(self, features: dict) -> tuple[float, bool]:
        """Return (probability, flagged) for a single product's feature dict.

        Builds a 1-row frame in the model's expected column order, then scores.
        Because preprocessing lives inside the Pipeline, we pass raw-ish features and
        the model handles imputation/encoding/scaling consistently with training.
        """
        cols = self.card.feature_names or list(features.keys())
        row = pd.DataFrame([{c: features.get(c) for c in cols}])
        proba = float(self.model.predict_proba(row)[:, 1][0])
        return proba, proba >= self.threshold
