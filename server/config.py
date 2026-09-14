"""Runtime configuration.

Every value is overridable by environment variable, so the same image runs as a
district server (Postgres, GPU) or as a PHC edge node (SQLite, CPU, offline)
with no code change.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRISHTI_", env_file=".env",
                                      extra="ignore")

    # --- identity -----------------------------------------------------------
    node_id: str = "district-1"
    node_role: str = "district"          # "district" | "edge"

    # --- database -----------------------------------------------------------
    # SQLite by default so the edge node and the test suite need no server.
    database_url: str = "sqlite:///./drishti.db"

    # --- auth ---------------------------------------------------------------
    # Refused at startup in district mode if left at the default (see check()).
    secret_key: str = "dev-only-insecure-change-me"
    access_token_minutes: int = 12 * 60      # one clinic shift
    bootstrap_admin_email: str | None = None
    bootstrap_admin_password: str | None = None

    # --- inference ----------------------------------------------------------
    model_dir: str = "artifacts"
    model_name: str = "grader.onnx"
    model_meta: str = "grader.json"
    device: str = "auto"                 # auto | cuda | cpu
    batch_size: int = 8
    batch_wait_ms: int = 25              # how long to coalesce a GPU batch
    inference_workers: int = 1
    image_size: int = 512

    # --- storage ------------------------------------------------------------
    storage_dir: str = "storage"
    max_upload_mb: int = 25

    # --- clinical safety ----------------------------------------------------
    # Referral decision cut-point. Deliberately separate from the model's own
    # grade thresholds so a district can tighten screening without retraining.
    referable_grade: int = 2
    low_confidence: float = 0.45

    @property
    def is_edge(self) -> bool:
        return self.node_role == "edge"

    def check(self):
        """Fail fast on configurations that are unsafe in production.

        A default signing key on a district server means anyone can mint a
        token for any role, including ophthalmologist sign-off. That must stop
        the process, not emit a warning nobody reads.
        """
        problems = []
        if not self.is_edge and self.secret_key == "dev-only-insecure-change-me":
            if os.environ.get("DRISHTI_ALLOW_INSECURE") != "1":
                problems.append(
                    "DRISHTI_SECRET_KEY is still the default. Set a real secret, "
                    "or export DRISHTI_ALLOW_INSECURE=1 for local development.")
        if self.access_token_minutes > 24 * 60:
            problems.append("access_token_minutes exceeds 24h; shorten it.")
        if problems:
            raise RuntimeError("unsafe configuration:\n  - " + "\n  - ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
