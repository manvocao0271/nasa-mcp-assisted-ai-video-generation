"""WILL.AI — What Infinity Looks Like AI

Streamlit multipage app entrypoint.
  Chat page      — pages/chat.py
  Video Studio   — pages/video_studio.py

Launch:
    uv run streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="WILL.AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.session import get_or_create_session_id, init_session_state, cleanup_expired_sessions
from utils.styles import inject_css

# ── Global CSS ─────────────────────────────────────────────────────────────────
inject_css()

# ── Session management ─────────────────────────────────────────────────────────
_sid = get_or_create_session_id()
init_session_state(_sid)

# Run cleanup once per session (removes dirs older than 24 h)
if not st.session_state.get("_cleanup_done"):
    cleanup_expired_sessions()
    st.session_state._cleanup_done = True

# ── API key guard ──────────────────────────────────────────────────────────────
if not os.environ.get("QWEN_API_KEY", ""):
    st.error("**QWEN_API_KEY is not set.** Add it to your .env file and restart.")
    st.stop()

# ── Navigation ─────────────────────────────────────────────────────────────────
_pg = st.navigation(
    [
        st.Page("pages/chat.py", title="Chat", default=True),
        st.Page("pages/video_studio.py", title="Video Studio"),
    ],
    position="hidden",
)
_pg.run()
