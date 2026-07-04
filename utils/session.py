"""Session management for anonymous multi-user isolation.

Each browser session gets a unique UUID stored in st.query_params["sid"].
All session data lives under output/sessions/{session_id}/.
Sessions older than max_age_hours are auto-cleaned at session init.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import streamlit as st


def get_or_create_session_id() -> str:
    """Return the current session ID, creating and persisting one if absent.

    Priority:
    1. ``st.session_state.session_id`` — survives page switches within a tab.
    2. ``st.query_params["sid"]`` — survives page refreshes (URL bookmark).
    3. Generate a new UUID and write it to both locations.
    """
    # 1. In-memory (page switch within same tab)
    if "session_id" in st.session_state:
        sid = st.session_state.session_id
        # Keep query param in sync so the URL stays shareable
        if st.query_params.get("sid") != sid:
            st.query_params["sid"] = sid
        return sid

    # 2. URL query param (page refresh / direct link)
    sid = st.query_params.get("sid", "")
    if sid and len(sid) == 36:  # basic UUID length check
        st.session_state.session_id = sid
        return sid

    # 3. Brand-new session
    sid = str(uuid.uuid4())
    st.session_state.session_id = sid
    st.query_params["sid"] = sid
    return sid


def get_session_dir(session_id: str) -> Path:
    """Return the data directory for *session_id*, creating it if necessary."""
    path = Path("output") / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_expired_sessions(max_age_hours: int = 24) -> None:
    """Remove session directories that have not been modified in *max_age_hours*."""
    sessions_root = Path("output") / "sessions"
    if not sessions_root.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            if session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
        except OSError:
            pass


def init_session_state(session_id: str) -> None:
    """Idempotently initialise all session state keys for *session_id*.

    Safe to call on every Streamlit re-run — only sets keys that are absent.
    """
    from agent.run_db import RunDB

    session_dir = get_session_dir(session_id)

    if "run_db" not in st.session_state:
        st.session_state.run_db = RunDB(db_path=session_dir / "runs.db")

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    if "session_id" not in st.session_state:
        st.session_state.session_id = session_id

    for key, default in [
        ("messages", []),
        ("video_queue", []),
        ("_pipeline_running", False),
        ("_pipeline_updates", []),
        ("_pipeline_queue", None),
        ("_pipeline_cancel", None),
        ("_pipeline_prompt", ""),
        ("_pipeline_chat_desc", ""),
        ("_pipeline_merged_assets", {}),
        ("_pipeline_run_saved", False),
        ("_cleanup_done", False),
        ("pending_prompt", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
