"""Synchronous HTTP client for the Rihla API (Streamlit scripts are synchronous)."""

from typing import Any

import httpx
import streamlit as st

import config


class APIError(Exception):
    """Request failed. kind is 'timeout', 'connection' or 'http' (non-2xx)."""

    def __init__(self, kind: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class TravelAIClient:
    """Thin wrapper over the API; every failure is raised as APIError so the UI never crashes."""

    def __init__(self, base_url: str, timeout: float = config.REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.TimeoutException as exc:
            raise APIError("timeout", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise APIError("connection", f"cannot reach the API ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise APIError("http", f"HTTP {response.status_code}", response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise APIError("http", "invalid JSON from the API", response.status_code) from exc

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        """POST /api/v1/chat."""
        return self._request("POST", "/api/v1/chat", json={"session_id": session_id, "message": message})  # type: ignore[no-any-return]

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """GET /api/v1/chat/{session_id}/history -> list of messages (oldest first)."""
        return self._request("GET", f"/api/v1/chat/{session_id}/history")["messages"]  # type: ignore[no-any-return]

    def get_analytics(self, days: int = 7) -> dict[str, Any]:
        """GET /api/v1/analytics."""
        key = config.get_analytics_api_key()
        headers = {"X-API-Key": key} if key else {}
        return self._request("GET", "/api/v1/analytics", params={"days": days}, headers=headers)  # type: ignore[no-any-return]

    def health_check(self) -> bool:
        """True when /health answers 200 with a connected database."""
        try:
            return bool(self._request("GET", "/health").get("db") == "connected")
        except APIError:
            return False


@st.cache_resource
def get_api_client() -> TravelAIClient:
    """One client per Streamlit server process."""
    return TravelAIClient(config.API_BASE_URL)
