"""Agency analytics dashboard."""

import hmac
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from utils.api_client import APIError, TravelAIClient
from utils.labels import INQUIRY_STATUS_LABELS, INQUIRY_TYPE_LABELS, intent_label, language_label

TUNISIA_TZ = timezone(timedelta(hours=1))
PRIMARY = "#0E7C86"
PALETTE = px.colors.qualitative.Safe


def _local(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(TUNISIA_TZ).strftime("%d/%m/%Y %H:%M")


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=13), legend=dict(orientation="h", y=-0.15), showlegend=True,
    )
    return fig


def _kpis(summary: dict[str, Any], days: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Conversations ({days} j)", summary["total_conversations"])
    c2.metric(f"Messages ({days} j)", summary["total_messages"])
    c3.metric("Taux d'automatisation", f"{summary['automation_rate']:.0%}",
              help="Part des conversations sans demande de vol/hôtel à rappeler (traitées sans conseiller).")
    c4.metric("Messages / conversation", summary["avg_messages_per_conversation"])


def _chart_days(data: dict[str, Any]) -> None:
    st.subheader("Conversations par jour")
    df = pd.DataFrame(data["conversations_by_day"]).rename(columns={"date": "Jour", "count": "Conversations"})
    fig = px.bar(df, x="Jour", y="Conversations", color_discrete_sequence=[PRIMARY])
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title=None, dtick=1 if df["Conversations"].max() < 8 else None)
    fig.update_layout(showlegend=False)
    st.plotly_chart(_style(fig), width="stretch")


def _chart_languages(data: dict[str, Any]) -> None:
    st.subheader("Répartition des langues")
    rows = data["language_distribution"]
    if not rows:
        st.info("Aucune conversation sur la période.")
        return
    df = pd.DataFrame({"Langue": [language_label(r["language"]) for r in rows], "Conversations": [r["count"] for r in rows]})
    fig = px.pie(df, names="Langue", values="Conversations", hole=0.45, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+value", sort=False)
    st.plotly_chart(_style(fig), width="stretch")


def _chart_intents(data: dict[str, Any]) -> None:
    st.subheader("Intentions les plus fréquentes")
    rows = data["intent_distribution"]
    if not rows:
        st.info("Aucun message sur la période.")
        return
    df = pd.DataFrame({"Intention": [intent_label(r["intent"]) for r in rows], "Messages": [r["count"] for r in rows]})
    fig = px.bar(df.iloc[::-1], x="Messages", y="Intention", orientation="h", color_discrete_sequence=[PRIMARY])
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    fig.update_layout(showlegend=False)
    st.plotly_chart(_style(fig), width="stretch")


def _table_inquiries(data: dict[str, Any]) -> None:
    st.subheader("Demandes à rappeler")
    rows = data["recent_inquiries"]
    if not rows:
        st.info("Aucune demande de vol ou d'hôtel pour le moment.")
        return
    df = pd.DataFrame({
        "Type": [INQUIRY_TYPE_LABELS.get(r["inquiry_type"], r["inquiry_type"]) for r in rows],
        "Destination": [r["destination"] or "—" for r in rows],
        "Message du client": [r["raw_message"] for r in rows],
        "Reçue le": [_local(r["created_at"]) for r in rows],
        "Statut": [INQUIRY_STATUS_LABELS.get(r["status"], r["status"]) for r in rows],
    })
    st.dataframe(df, hide_index=True, width="stretch")


def _table_conversations(data: dict[str, Any]) -> None:
    st.subheader("Dernières conversations")
    rows = data["recent_conversations"]
    if not rows:
        st.info("Aucune conversation enregistrée.")
        return
    df = pd.DataFrame({
        "Session": [r["session_id"] for r in rows],
        "Langue": [language_label(r["language"]) for r in rows],
        "Messages": [r["message_count"] for r in rows],
        "Premier message": [r["first_message"] for r in rows],
        "Début": [_local(r["created_at"]) for r in rows],
    })
    st.dataframe(df, hide_index=True, width="stretch")


def _body(client: TravelAIClient, days: int) -> None:
    try:
        data = client.get_analytics(days)
    except APIError as error:
        if error.status_code == 401:
            st.error("Accès refusé par l'API : la clé ANALYTICS_API_KEY est absente ou incorrecte.")
        else:
            st.error("Impossible de charger les statistiques. Vérifiez que l'API est démarrée.")
        return
    _kpis(data["summary"], days)
    st.divider()
    left, right = st.columns(2)
    with left:
        _chart_days(data)
    with right:
        _chart_languages(data)
    left, right = st.columns(2)
    with left:
        _chart_intents(data)
    with right:
        _table_inquiries(data)
    st.divider()
    _table_conversations(data)
    st.caption("Dernière mise à jour : " + datetime.now(TUNISIA_TZ).strftime("%d/%m/%Y %H:%M:%S"))


def _login_gate() -> bool:
    """Password screen. True once the visitor is authenticated (kept in the browser session only)."""
    state = st.session_state
    if state.get("dashboard_authenticated"):
        return True
    st.title("🔐 Accès tableau de bord")
    expected = config.get_dashboard_password()
    if not expected:
        st.error("Le mot de passe du tableau de bord n'est pas configuré (variable DASHBOARD_PASSWORD).")
        return False
    with st.form("dashboard_login"):  # a form lets Enter submit
        entered = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            state.dashboard_authenticated = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")
    return False


def render(client: TravelAIClient) -> None:
    """Draw the dashboard behind the password gate. Data is fetched on every run (no caching)."""
    if not _login_gate():
        return
    if st.sidebar.button("🔓 Déconnexion"):
        st.session_state.dashboard_authenticated = False
        st.rerun()
    st.title("📊 Tableau de bord")
    controls, _ = st.columns([2, 3])
    with controls:
        days = st.selectbox("Période", [7, 14, 30], format_func=lambda d: f"{d} derniers jours")
        auto = st.toggle(f"Actualisation automatique ({config.REFRESH_INTERVAL_SECONDS} s)", value=False)
        st.button("🔄 Actualiser")  # a click reruns the script, which refetches
    body = st.fragment(run_every=config.REFRESH_INTERVAL_SECONDS if auto else None)(_body)
    body(client, days)
