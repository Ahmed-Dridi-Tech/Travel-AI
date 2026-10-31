"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration. Secrets default to empty and are never hardcoded."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # App
    app_name: str = "travel-ai"
    app_env: Literal["development", "test", "production"] = "development"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Database
    postgres_user: str = "travel"
    postgres_password: str = ""
    postgres_db: str = "travel_ai"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = ""

    # LLM
    llm_provider: Literal["mock", "anthropic", "gemini"] = "mock"
    llm_timeout_seconds: float = 20.0
    llm_max_tool_iterations: int = Field(default=4, ge=1, le=10)
    llm_max_tokens: int = Field(default=500, ge=50, le=4000)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)  # not sent to Anthropic (rejected since 2026-10-21)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # External APIs
    openweather_api_key: str = ""
    openweather_base_url: str = "https://api.openweathermap.org"
    exchange_rate_api_key: str = ""
    exchangerate_base_url: str = "https://v6.exchangerate-api.com/v6"
    http_timeout_seconds: float = 8.0
    cache_ttl_weather_seconds: int = 900
    cache_ttl_exchange_seconds: int = 3600

    # NLP
    fasttext_model_path: str = "data/models/lid.176.bin"
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    intent_model_path: str = "data/models/intent_classifier.pkl"
    intent_labels_path: str = "data/models/intent_labels.pkl"
    intent_config_path: str = "data/models/intent_config.json"
    gazetteer_path: str = "data/destinations_gazetteer.json"
    spacy_model_name: str = "xx_ent_wiki_sm"
    intent_confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    ood_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # Frontend
    api_base_url: str = "http://localhost:8000"
    dashboard_password: str = ""  # empty = dashboard locked (no built-in default password)

    # Deployment
    use_mock_nlp: bool = False  # True on the lean (no-ML) deployment: keyword-based MockNLPPipeline
    analytics_api_key: str = ""  # when set, GET /api/v1/analytics requires header X-API-Key

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string (the .env format) as well as a list."""
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _build_database_url(self) -> "Settings":
        """Compose DATABASE_URL from POSTGRES_* parts unless one was provided explicitly."""
        for prefix in ("postgres://", "postgresql://"):  # hosts (Railway, Heroku) hand out the plain scheme
            if self.database_url.startswith(prefix):
                self.database_url = "postgresql+psycopg://" + self.database_url[len(prefix):]
        if not self.database_url:
            self.database_url = (
                f"postgresql+psycopg://{quote_plus(self.postgres_user)}:"
                f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def is_production(self) -> bool:
        """True when running with APP_ENV=production."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance (override in tests via cache_clear)."""
    return Settings()
