"""Rihla web app: customer chat + agency dashboard.  Run: streamlit run frontend/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat imports: config, utils, views

import streamlit as st  # noqa: E402

import config  # noqa: E402
from utils.api_client import get_api_client  # noqa: E402
from views import chat, dashboard  # noqa: E402

st.set_page_config(page_title=config.PAGE_TITLE, page_icon="🧳", layout="wide")

PAGES = {"💬 Assistant Rihla": chat.render, "📊 Tableau de bord": dashboard.render}

client = get_api_client()
with st.sidebar:
    st.markdown("## 🧳 Rihla")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed", key="page")
    st.divider()

if not client.health_check():
    st.warning(f"L'API est injoignable ou la base de données est indisponible ({config.API_BASE_URL}).")

PAGES[page](client)
