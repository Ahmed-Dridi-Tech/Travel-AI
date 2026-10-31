"""Railway deployment files, and a proof that the lean install really works without any ML package."""

import json
import os
import re
import subprocess
import sys
import textwrap

from app.nlp.intent_classifier import PROJECT_ROOT


def _names(lines: list[str]) -> set[str]:
    out = set()
    for line in lines:
        line = line.split("#")[0].strip().strip('",')
        if line:
            out.add(re.split(r"[\[<>=!~ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return out


def test_root_requirements_match_pyproject_dependencies() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M).group(1)  # type: ignore[union-attr]
    declared = _names(block.splitlines())
    pinned = _names((PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert declared == pinned, (declared ^ pinned)


def test_requirements_pin_versions_and_exclude_heavy_packages() -> None:
    lines = [l for l in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert all("==" in l for l in lines)
    heavy = {"torch", "sentence-transformers", "spacy", "fasttext-wheel", "fasttext", "scikit-learn", "numpy", "joblib"}
    assert not heavy & _names(lines)


def test_frontend_requirements() -> None:
    names = _names((PROJECT_ROOT / "frontend" / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert {"streamlit", "httpx", "plotly", "pandas"} <= names


def test_railway_config() -> None:
    config = json.loads((PROJECT_ROOT / "railway.json").read_text(encoding="utf-8"))
    deploy = config["deploy"]
    assert config["build"]["builder"] == "NIXPACKS"
    assert deploy["healthcheckPath"] == "/health"
    assert deploy["preDeployCommand"] == ["alembic upgrade head"]  # Railway has no Procfile "release" phase
    assert "$PORT" in deploy["startCommand"] and "app.main:app" in deploy["startCommand"]


def test_frontend_railway_config_is_self_contained() -> None:
    config = json.loads((PROJECT_ROOT / "frontend" / "railway.json").read_text(encoding="utf-8"))["deploy"]
    assert config["startCommand"].startswith("streamlit run streamlit_app.py") and "$PORT" in config["startCommand"]
    assert "alembic" not in json.dumps(config)  # the Streamlit service must never run the API's migration
    assert config["healthcheckPath"] == "/_stcore/health"


def test_procfiles_and_runtime() -> None:
    api = (PROJECT_ROOT / "Procfile").read_text(encoding="utf-8")
    assert api.startswith("web: uvicorn app.main:app") and "--port $PORT" in api and "release:" not in api
    ui = (PROJECT_ROOT / "frontend" / "Procfile").read_text(encoding="utf-8")
    assert "streamlit run streamlit_app.py" in ui and "--server.port $PORT" in ui and "--server.address 0.0.0.0" in ui
    assert (PROJECT_ROOT / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.10.11"


def test_env_example_documents_deployment_variables() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("USE_MOCK_NLP=false", "ANALYTICS_API_KEY=", "DASHBOARD_PASSWORD="):
        assert key in text
    assert "DASHBOARD_PASSWORD=change-this-before-demo" in text  # only a placeholder, never a real secret


LEAN_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sys

    BLOCKED = {"numpy", "scipy", "sklearn", "torch", "transformers", "sentence_transformers", "spacy",
               "fasttext", "joblib", "pandas"}

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked heavy package: " + name)

    sys.meta_path.insert(0, Blocker())

    from app.main import create_app
    from app.agent.orchestrator import AgentOrchestrator
    from app.nlp import get_pipeline
    from app.nlp.mock_pipeline import MockNLPPipeline

    create_app()
    pipeline = get_pipeline()
    assert isinstance(pipeline, MockNLPPipeline), type(pipeline)
    result = pipeline.analyze("météo Djerba")
    assert (result.intent, result.destinations) == ("weather", ["djerba"]), result
    AgentOrchestrator()  # builds mock NLP + tool registry + mock LLM
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("LEAN-OK")
    """
)


def test_app_boots_and_analyzes_with_every_ml_package_blocked() -> None:
    env = {**os.environ, "USE_MOCK_NLP": "true", "LLM_PROVIDER": "mock", "PYTHONIOENCODING": "utf-8"}
    run = subprocess.run([sys.executable, "-c", LEAN_SCRIPT], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0 and "LEAN-OK" in run.stdout, run.stderr[-2000:]
