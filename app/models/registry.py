"""Model registry: persist a model together with its ModelCard.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    A model artifact without its context is a liability. The card records what the
    model is, what data window built it, which features it expects (the serving
    contract), and the metrics it shipped with — enough to audit or roll back.

KEY CONCEPT — the card is the train/serve contract
    Serving (app/serving/predictor.py) builds its input row from
    card.feature_names, in order. Rename or reorder features at training time and
    the card — not a crash — is what keeps serving consistent.

TEST
    exercised indirectly via scripts/train.py and the serving predictor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ModelCard:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    data_window: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)


def save_model(model: Any, card: ModelCard, path: Path) -> Path:
    """Persist model+card as one joblib artifact, plus a human-readable JSON sidecar.

    One artifact (not two files loaded separately) so model and card can never get
    out of sync; the .card.json sidecar exists only for humans and diff tools.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "card": card}, path)
    sidecar = path.with_suffix(".card.json")
    sidecar.write_text(json.dumps(asdict(card), indent=2, default=str), encoding="utf-8")
    return path


def load_model(path: Path) -> tuple[Any, ModelCard]:
    """Load (model, card) saved by save_model. Fails loud on a malformed artifact."""
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or "model" not in payload or "card" not in payload:
        raise ValueError(f"{path} is not a registry artifact (expected {{model, card}})")
    return payload["model"], payload["card"]
