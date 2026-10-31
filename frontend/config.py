"""Frontend settings (environment variables override the defaults)."""

import os
from pathlib import Path

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
PAGE_TITLE: str = "Rihla — Assistant Voyage Tunisie"
REFRESH_INTERVAL_SECONDS: int = int(os.getenv("REFRESH_INTERVAL_SECONDS", "30"))
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))


_DOTENV = Path(__file__).resolve().parents[1] / ".env"


def _dotenv_value(key: str) -> str:
    """Value of KEY in the project's .env ('' if absent). Minimal parser: KEY=VALUE, optional quotes."""
    try:
        lines = _DOTENV.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            value = value.strip()
            if value[:1] in {"'", '"'} and value[-1:] == value[:1] and len(value) > 1:
                return value[1:-1]
            return value.split(" #")[0].strip()
    return ""


def get_dashboard_password() -> str:
    """DASHBOARD_PASSWORD from the environment, else .env; empty means the dashboard stays locked."""
    return os.environ.get("DASHBOARD_PASSWORD") or _dotenv_value("DASHBOARD_PASSWORD")


def get_analytics_api_key() -> str:
    """ANALYTICS_API_KEY from the environment, else .env; empty means the API is called without a key."""
    return os.environ.get("ANALYTICS_API_KEY") or _dotenv_value("ANALYTICS_API_KEY")
