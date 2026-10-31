import pytest

from app.core.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DATABASE_URL", "POSTGRES_PASSWORD", "POSTGRES_HOST", "CORS_ORIGINS", "LOG_LEVEL"):
        monkeypatch.delenv(key, raising=False)


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_database_url_built_from_parts() -> None:
    s = make(postgres_user="u", postgres_password="p@ss", postgres_host="db", postgres_db="x")
    assert s.database_url == "postgresql+psycopg://u:p%40ss@db:5432/x"


def test_explicit_database_url_wins() -> None:
    s = make(database_url="postgresql+psycopg://a:b@h:1/d")
    assert s.database_url == "postgresql+psycopg://a:b@h:1/d"


def test_cors_origins_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    assert make().cors_origins == ["http://a.com", "http://b.com"]


def test_log_level_uppercased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert make().log_level == "DEBUG"


def test_defaults_have_no_secrets() -> None:
    s = make()
    assert s.anthropic_api_key == "" and s.openweather_api_key == ""
    assert s.llm_provider == "mock"


@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://"])
def test_hosted_database_url_scheme_is_normalised_for_psycopg(scheme: str) -> None:
    s = make(database_url=f"{scheme}u:p@host.railway.internal:5432/railway")
    assert s.database_url == "postgresql+psycopg://u:p@host.railway.internal:5432/railway"
    assert make(database_url="postgresql+psycopg://a:b@h:1/d").database_url == "postgresql+psycopg://a:b@h:1/d"


def test_deploy_flags_default_safe() -> None:
    s = make()
    assert s.use_mock_nlp is False and s.analytics_api_key == "" and s.dashboard_password == ""
    assert make(use_mock_nlp=True).use_mock_nlp is True
