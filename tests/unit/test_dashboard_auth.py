"""Password gate on the dashboard, driven through Streamlit's headless test harness."""

import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.nlp.intent_classifier import PROJECT_ROOT  # noqa: E402

FRONTEND = PROJECT_ROOT / "frontend"
APP = str(FRONTEND / "streamlit_app.py")
PASSWORD = "S3cret-for-tests"


@pytest.fixture(autouse=True)
def _frontend_on_path() -> None:
    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))


def open_dashboard() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("📊 Tableau de bord").run()
    return at


def submit(at: AppTest, value: str) -> AppTest:
    at.text_input[0].input(value)  # no run() here: a rerun before submitting would reset the form field
    next(b for b in at.button if b.label == "Se connecter").click().run()
    return at


def authenticated(at: AppTest) -> bool:
    """On screen: the dashboard title is shown instead of the login screen."""
    return at.title[0].value.startswith("📊")


def is_password_field(at: AppTest) -> bool:
    return len(at.text_input) == 1 and at.text_input[0].proto.type == 1  # TextInput.PASSWORD


def test_dashboard_requires_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    at = open_dashboard()
    assert not at.exception
    assert at.title[0].value.endswith("Accès tableau de bord")
    assert is_password_field(at)
    assert len(at.metric) == 0 and not authenticated(at)


def test_wrong_password_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    at = submit(open_dashboard(), "not-the-password")
    assert [e.value for e in at.error] == ["Mot de passe incorrect"]
    assert len(at.metric) == 0 and not authenticated(at)


def test_correct_password_unlocks_and_logout_relocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    at = submit(open_dashboard(), PASSWORD)
    assert authenticated(at)
    assert not at.text_input  # the password box is gone
    next(b for b in at.sidebar.button if b.label == "🔓 Déconnexion").click().run()
    assert not authenticated(at)
    assert is_password_field(at)


def test_dashboard_stays_locked_when_no_password_is_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import config as frontend_config

    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr(frontend_config, "_DOTENV", tmp_path / "missing.env")
    at = open_dashboard()
    assert not at.text_input and not authenticated(at)
    assert "DASHBOARD_PASSWORD" in at.error[0].value


def test_chat_page_needs_no_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_PASSWORD", PASSWORD)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and at.title[0].value.startswith("Rihla")


def test_dotenv_fallback_parser(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import config as frontend_config

    env = tmp_path / ".env"
    env.write_text('A=1\nDASHBOARD_PASSWORD="pa ss#word"  \nOTHER=x\n', encoding="utf-8")
    monkeypatch.setattr(frontend_config, "_DOTENV", env)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    assert frontend_config.get_dashboard_password() == "pa ss#word"
    env.write_text("DASHBOARD_PASSWORD=plain # comment\n", encoding="utf-8")
    assert frontend_config.get_dashboard_password() == "plain"
    monkeypatch.setenv("DASHBOARD_PASSWORD", "from-env")
    assert frontend_config.get_dashboard_password() == "from-env"  # env wins over .env
