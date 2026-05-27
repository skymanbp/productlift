"""Configuration. FULLY BUILT — the single source of truth.

Two layers:
  - Settings (env): paths and secrets, from .env / environment. Things that change
    per machine.
  - params (config/params.yaml): the experiment definition — target, split dates,
    feature/model/eval knobs. Things that change per experiment and that you want
    version-controlled so a result is reproducible.

Import `get_settings()` and `get_params()` everywhere; never read os.environ or
re-parse yaml ad hoc.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "config" / "params.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "current.joblib"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    random_seed: int = 42

    kaggle_username: str = ""
    kaggle_key: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_params() -> dict[str, Any]:
    """Load config/params.yaml as a plain dict (cached)."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
