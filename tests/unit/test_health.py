from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from app.database.session import get_db
from app.main import create_app


class _FakeSession:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    async def execute(self, *_: Any, **__: Any) -> None:
        if self.fail:
            raise ConnectionError("db down")


def _client(fail: bool) -> TestClient:
    app = create_app()

    async def fake_db() -> AsyncIterator[_FakeSession]:
        yield _FakeSession(fail)

    app.dependency_overrides[get_db] = fake_db
    return TestClient(app)


def test_health_ok() -> None:
    resp = _client(fail=False).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["db"] == "connected"
    assert {"app", "version", "env"} <= body.keys()


def test_health_degraded_returns_200() -> None:
    resp = _client(fail=True).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded" and resp.json()["db"] == "unreachable"
