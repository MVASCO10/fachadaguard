"""Configuração central lida de variáveis de ambiente (ver .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FG_", extra="ignore")

    project_name: str = "FachadaGuard"
    environment: str = Field(default="demo", pattern="^(demo|dev|test|prod)$")

    database_url: str = "postgresql+psycopg://fachadaguard:fachadaguard_demo@localhost:55432/fachadaguard"

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = "fg_ingestor"
    mqtt_password: str = ""
    mqtt_tls: bool = False
    mqtt_ca_file: str | None = None
    mqtt_topic_prefix: str = "fachadaguard/v1"
    mqtt_qos: int = Field(default=1, ge=0, le=2)
    mqtt_client_id: str = "fg-ingestor"

    api_base_url: str = "http://localhost:8000"
    api_token: str = ""

    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    max_upload_mb: float = Field(default=15.0, gt=0, le=100)

    worker_poll_seconds: float = Field(default=1.0, gt=0)
    export_interval_seconds: float = Field(default=300.0, gt=0)
    random_seed: int = 42

    genai_enabled: bool = False
    anthropic_api_key: str = ""
    genai_model: str = "claude-opus-5-5"

    @field_validator("mqtt_topic_prefix")
    @classmethod
    def _no_wildcards(cls, value: str) -> str:
        if any(ch in value for ch in "#+") or value.endswith("/"):
            raise ValueError("prefixo de tópico não pode conter curingas nem terminar com '/'")
        return value

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def lake_dir(self) -> Path:
        return self.data_dir / "lake"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    def safe_dict(self) -> dict[str, object]:
        """Configuração sem segredos, adequada para logs e /health."""
        hidden = {"mqtt_password", "anthropic_api_key", "api_token", "database_url"}
        data = {k: v for k, v in self.model_dump().items() if k not in hidden}
        data["database_url"] = self.database_url.split("@")[-1]
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
