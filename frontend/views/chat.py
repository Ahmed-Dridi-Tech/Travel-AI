"""Customer chat page."""

import html
import re
import uuid

import streamlit as st

from utils.api_client import APIError, TravelAIClient
from utils.labels import INTENT_LABELS, LANGUAGE_LABELS, intent_label, language_label

__all__ = ["INTENT_LABELS", "LANGUAGE_LABELS", "render"]

_ARABIC = re.compile("[؀-ۿ]")
STATE_KEYS = ("session_id", "messages", "history_loaded", "last_language", "last_intent")


def _show(text: str) -> None:
    """Render a message; Arabic-majority text is right-aligned (escaped first: content is untrusted)."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if _ARABIC.match(c)) / len(letters) > 0.5:
        safe = html.escape(text).replace("\n", "<br>")
        st.markdown(f'<div dir="rtl" style="text-align:right">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def _load_history(client: TravelAIClient) -> None:
    state = st.session_state
    try:
        history = client.get_history(state.session_id)
    except APIError:
        history = []  # a new session has none; an outage is reported on the first send
    state.messages = [{"role": m["role"], "content": m["content"]} for m in history]
    if history:
        last_user = next((m for m in reversed(history) if m["role"] == "user"), None)
        state.last_intent = last_user["intent"] if last_user else None
    state.history_loaded = True


def _error_text(error: APIError) -> str:
    if error.kind == "timeout":
        return "La réponse prend plus de temps que prévu. Veuillez réessayer dans un instant."
    return "Service temporairement indisponible. Veuillez réessayer."


def _sidebar() -> None:
    state = st.session_state
    with st.sidebar:
        st.markdown("### Conversation")
        st.write("**Langue détectée** : " + language_label(state.get("last_language") or "unknown"))
        st.write("**Intention** : " + (intent_label(state.last_intent) if state.get("last_intent") else "—"))
        st.caption(f"Session : {state.session_id}")
        if st.button("🗑️ Effacer la conversation", width="stretch"):
            for key in STATE_KEYS:
                state.pop(key, None)
            st.rerun()


def render(client: TravelAIClient) -> None:
    """Draw the chat page and handle one user turn."""
    state = st.session_state
    state.setdefault("session_id", str(uuid.uuid4()))
    state.setdefault("messages", [])
    if not state.get("history_loaded"):
        _load_history(client)

    st.title("Rihla — Assistant voyage Tunisie 🇹🇳")
    st.caption("Posez vos questions en français, anglais, arabe ou dialecte tunisien")

    for message in state.messages:
        with st.chat_message(message["role"]):
            _show(message["content"])

    prompt = st.chat_input("Votre message...")
    if prompt and prompt.strip():
        prompt = prompt.strip()
        state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            _show(prompt)  # optimistic: shown before the API answers
        with st.chat_message("assistant"):
            try:
                with st.spinner("Rihla réfléchit..."):
                    data = client.send_message(state.session_id, prompt)
            except APIError as error:
                st.error(_error_text(error))
            else:
                _show(data["response"])
                state.messages.append({"role": "assistant", "content": data["response"]})
                state.last_language = data.get("language")
                state.last_intent = data.get("intent")
    _sidebar()
