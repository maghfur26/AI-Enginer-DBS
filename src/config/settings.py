"""
AE2 — Configuration
Centralized config menggunakan pydantic-settings.
Semua environment variable dikelola di sini.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────
    app_name: str = "OLAH Inference API"
    app_version: str = "1.0.0"
    team_id: str = "CC26-PSU127"
    debug: bool = False

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2

    # ── Model Artifacts (dari AE1 handoff) ────────────────────────────────
    artifacts_dir: str = "artifacts"
    model_filename: str = "model.pt"
    tokenizer_dir: str = "tokenizer"
    label_map_filename: str = "label_map.json"
    vocab_filename: str = "vocab.json"

    # ── Inference ─────────────────────────────────────────────────────────
    max_sequence_length: int = 64
    batch_size_max: int = 50        # Batas batch per request
    top_n_default: int = 5          # Default top-N rekomendasi
    min_score_default: float = 0.3  # Skor minimum rekomendasi

    # ── Cache (Redis) ─────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 86400  # 1 hari

    # ── CORS ──────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def model_path(self) -> str:
        import os
        return os.path.join(self.artifacts_dir, self.model_filename)

    @property
    def tokenizer_path(self) -> str:
        import os
        return os.path.join(self.artifacts_dir, self.tokenizer_dir)

    @property
    def label_map_path(self) -> str:
        import os
        return os.path.join(self.artifacts_dir, self.label_map_filename)


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
