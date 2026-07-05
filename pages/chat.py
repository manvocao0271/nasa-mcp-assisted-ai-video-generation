"""Chat page + integrated Video Studio side panel — WILL.AI."""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from openai import AuthenticationError, APIError

from agent.chat_agent import ChatAgent
from agent.orchestrator import Orchestrator
from agent.qwen_client import QwenClient, MODEL_CHAT
from agent.run_db import Message
from utils.helpers import fetch_thumb
from utils.session import get_session_dir

# ── Environment ────────────────────────────────────────────────────────────────
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

# Session state already initialised by app.py
_db = st.session_state.run_db
_session_dir = get_session_dir(st.session_state.session_id)

# ── Suggested prompts ──────────────────────────────────────────────────────────
_SUGGESTIONS = [
    "What's today's Astronomy Picture of the Day?",
    "Show me the latest solar flare activity",
    "Find exoplanets in the habitable zone",
    "What asteroids are passing near Earth this week?",
]

# ── Pipeline stage definitions ─────────────────────────────────────────────────
_PIPELINE_STAGES = [
    ("script",     "📝", "Script"),
    ("storyboard", "🎨", "Storyboard"),
    ("video",      "🎬", "Video"),
]
_STAGE_DETAIL_LABELS = {
    "data":       "Fetching NASA data",
    "script":     "Writing scene captions",
    "storyboard": "Generating storyboard",
    "video":      "Generating video clips (Wan 2.7)",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _queue_image(img: dict, description: str, data_context: dict) -> None:
    """Append an image to the video queue, deduplicating by URL."""
    url = img.get("url", "")
    if not url:
        return
    if any(item["url"] == url for item in st.session_state.video_queue):
        return
    st.session_state.video_queue.append({
        "url": url,
        "thumb_url": img.get("thumb_url", ""),
        "caption": img.get("caption", ""),
        "source": img.get("source", "NASA"),
        "data_context": data_context,
        "description": description,
    })


def _save_chat_turn(
    user_message: str,
    assistant_response: str,
    chat_assets: dict | None = None,
    retrieved_passages: list | None = None,
) -> None:
    """Persist a chat turn to RunDB, including images and retrieved sources."""
    conv_id = st.session_state.conversation_id
    history_before = _db.get_conversation_history(conv_id)
    is_first = len(history_before) == 0
    _assets = {
        "images": (chat_assets or {}).get("images", []),
        "data":   (chat_assets or {}).get("data", {}),
        "_retrieved": retrieved_passages or [],
    }
    _db.save_run(
        run_id=str(uuid.uuid4()),
        conversation_id=conv_id,
        user_message=user_message,
        assistant_response=assistant_response,
        assets=_assets,
        manifest={},
        messages=[
            Message(role=m["role"], content=m["content"], timestamp=datetime.now().isoformat())
            for m in st.session_state.messages
        ],
    )
    if is_first:
        _db.set_conversation_title(conv_id, user_message[:120])


def _messages_from_history(history: list[dict]) -> list[dict]:
    """Convert RunDB history rows back into session-state message dicts."""
    msgs: list[dict] = []
    for run in history:
        msgs.append({"role": "user", "content": run["user_message"]})
        raw = dict(run.get("assets") or {})
        retrieved = raw.pop("_retrieved", [])
        msgs.append({
            "role": "assistant",
            "content": run["assistant_response"],
            "assets": raw,
            "retrieved_passages": retrieved,
        })
    return msgs


def _merge_data_contexts(queue_items: list[dict]) -> dict:
    """Merge data_context dicts from all queued items; later items win."""
    merged: dict = {}
    for item in queue_items:
        merged.update(item.get("data_context") or {})
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_stage_statuses(updates: list[dict]) -> dict[str, str]:
    """Return {stage_key: 'pending'|'active'|'done'} for the three display stages."""
    last_per_stage: dict[str, dict] = {}
    for upd in updates:
        s = upd.get("stage", "")
        if s:
            last_per_stage[s] = upd
    result: dict[str, str] = {}
    for key, _icon, _label in _PIPELINE_STAGES:
        upd = last_per_stage.get(key)
        if upd is None:
            result[key] = "pending"
        elif upd.get("status") == "running":
            result[key] = "active"
        else:
            result[key] = "done"
    return result


def _pipeline_steps_html(statuses: dict[str, str]) -> str:
    """Build the three-step pipeline row as inline HTML."""
    parts: list[str] = []
    for i, (key, icon, label) in enumerate(_PIPELINE_STAGES):
        status = statuses.get(key, "pending")
        if status == "done":
            cls, badge, badge_cls = "step-done", "✓", "badge-done"
        elif status == "active":
            cls, badge, badge_cls = "step-active", "●", "badge-active"
        else:
            cls, badge, badge_cls = "step-pending", "○", "badge-pending"
        parts.append(
            f'<div class="pipeline-step {cls}">'
            f'<div class="step-icon">{icon}</div>'
            f'<div class="step-name">{label}</div>'
            f'<div class="step-badge {badge_cls}">{badge}</div>'
            f"</div>"
        )
        if i < len(_PIPELINE_STAGES) - 1:
            parts.append('<div class="pipeline-arrow">→</div>')
    return f'<div class="pipeline-row">{"".join(parts)}</div>'


def _pipeline_log_html(updates: list[dict]) -> str:
    """Build a detailed pipeline execution log as HTML from all update dicts."""
    if not updates:
        return ""

    _STAGE_META: dict[str, tuple[str, str]] = {
        "script":     ("📝", "Script"),
        "storyboard": ("🎨", "Storyboard"),
        "video":      ("🎬", "Video"),
        "data":       ("🔭", "Data"),
    }

    stage_updates: dict[str, list[dict]] = {}
    stage_order: list[str] = []
    warnings: list[str] = []

    for upd in updates:
        stage = upd.get("stage", "")
        if stage == "warning":
            warnings.append(upd.get("detail", ""))
        elif stage and stage not in ("error", "done"):
            if stage not in stage_updates:
                stage_updates[stage] = []
                stage_order.append(stage)
            stage_updates[stage].append(upd)

    html: list[str] = ['<div class="pipeline-log">']

    for idx, stage in enumerate(stage_order):
        entries = stage_updates[stage]
        icon, label = _STAGE_META.get(stage, ("●", stage.title()))

        model     = next((e.get("model")      for e in entries if e.get("model")),      None)
        mode      = next((e.get("mode")       for e in entries if e.get("mode")),       None)
        resolution= next((e.get("resolution") for e in entries if e.get("resolution")), None)
        duration  = next((e.get("duration")   for e in entries if e.get("duration")),   None)

        meta_parts: list[str] = []
        if model:
            meta_parts.append(model)
        if mode:
            meta_parts.append(mode)
        if resolution:
            meta_parts.append(resolution)
        if duration:
            meta_parts.append(f"{duration}s")
        meta_str = " · ".join(meta_parts)

        is_done    = any(e.get("status") == "done"    for e in entries)
        is_running = not is_done and any(e.get("status") == "running" for e in entries)

        if is_done:
            status_badge = '<span class="log-status-done">✓</span>'
        elif is_running:
            status_badge = '<span class="log-status-running">●</span>'
        else:
            status_badge = '<span class="log-status-pending">○</span>'

        sep_cls = " log-stage-sep" if idx > 0 else ""
        html.append(
            f'<div class="log-stage">'
            f'<div class="log-stage-header{sep_cls}">'
            f'<span class="log-icon">{icon}</span>'
            f'<span class="log-label">{label}</span>'
        )
        if meta_str:
            html.append(f'<span class="log-model">{meta_str}</span>')
        html.append(f'{status_badge}</div><div class="log-entries">')

        for entry in entries:
            detail = entry.get("detail", "")
            status = entry.get("status", "")
            if not detail:
                continue
            if status == "done":
                html.append(f'<div class="log-entry log-done">✓&nbsp;{detail}</div>')
                for scene in entry.get("scenes", [])[:2]:
                    cap = scene.get("caption", "")[:70]
                    sc_n = scene.get("scene", "")
                    if cap:
                        html.append(
                            f'<div class="log-scene">Scene {sc_n}: {cap}…</div>'
                        )
            else:
                html.append(f'<div class="log-entry log-running">→&nbsp;{detail}</div>')

        html.append("</div></div>")

    for warn in warnings:
        html.append(f'<div class="log-warning">⚠&nbsp;{warn}</div>')

    html.append("</div>")
    return "".join(html)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO STUDIO PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _launch_pipeline(prompt_text: str, use_i2v: bool) -> None:
    """Build merged_assets and start the background pipeline thread."""
    _q = st.session_state.video_queue
    if use_i2v and _q:
        merged_assets: dict = {
            "query": prompt_text,
            "images": [
                {"url": i["url"], "thumb_url": i.get("thumb_url", ""),
                 "caption": i.get("caption", ""), "source": i.get("source", "NASA")}
                for i in _q
            ],
            "data": _merge_data_contexts(_q),
            "tools_called": [],
        }
        chat_desc = " ".join(i.get("description", "") for i in _q)[:400]
    else:
        _orch_fetch = Orchestrator(
            qwen_api_key=QWEN_API_KEY,
            nasa_api_key=NASA_API_KEY,
            output_dir=_session_dir,
        )
        merged_assets = {}
        _ph = st.empty()
        _ph.caption("Fetching NASA data…")
        try:
            for _upd in _orch_fetch.fetch_data(prompt_text):
                merged_assets = _upd.get("assets", merged_assets)
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            _ph.empty()
            return
        _ph.empty()
        chat_desc = ""

    st.session_state._pipeline_prompt = prompt_text
    st.session_state._pipeline_chat_desc = chat_desc
    st.session_state._pipeline_merged_assets = merged_assets
    st.session_state._pipeline_run_saved = False

    cancel_ev = threading.Event()
    pq: queue.Queue = queue.Queue()
    st.session_state._pipeline_cancel = cancel_ev
    st.session_state._pipeline_queue = pq
    st.session_state._pipeline_running = True
    st.session_state._pipeline_updates = []

    def _bg(
        _p=prompt_text, _a=merged_assets, _d=chat_desc,
        _q=pq, _ev=cancel_ev, _out=_session_dir,
    ) -> None:
        _orch = Orchestrator(
            qwen_api_key=QWEN_API_KEY, nasa_api_key=NASA_API_KEY,
            cancel_event=_ev, output_dir=_out,
        )
        try:
            for upd in _orch.run_pipeline(_p, _a, chat_description=_d):
                _q.put(upd)
        except InterruptedError:
            _q.put({"stage": "warning", "status": "warning", "detail": "Generation cancelled."})
        except Exception as exc:
            detail = str(exc)
            if "AllocationQuota" in detail or "FreeTier" in detail:
                detail = (
                    "Video generation quota exhausted. "
                    "Disable \"Use free tier only\" in the DashScope console."
                )
            _q.put({"stage": "error", "status": "error", "detail": detail})
        finally:
            _q.put(None)

    threading.Thread(target=_bg, daemon=True).start()


def _render_studio_panel() -> None:
    """Render the Video Studio side panel (called inside col_studio)."""
    _q = st.session_state.video_queue

    # ── Panel header ─────────────────────────────────────────────────────────
    _h, _x = st.columns([9, 1])
    with _h:
        st.markdown('<p class="studio-title">🎬 Video Studio</p>', unsafe_allow_html=True)
    with _x:
        if st.button("✕", key="close_studio", help="Close panel"):
            st.session_state.studio_open = False
            st.rerun()

    # ── INPUTS ───────────────────────────────────────────────────────────────
    st.markdown('<div class="studio-section">INPUTS</div>', unsafe_allow_html=True)

    if _q:
        for _qi, _item in enumerate(_q):
            _label = (_item.get("caption") or _item.get("source") or "Image").strip()
            if len(_label) > 58:
                _label = _label[:55] + "…"
            _ra, _rb = st.columns([9, 1])
            with _ra:
                st.markdown(
                    f'<div class="queue-label">🖼&nbsp;{_label}</div>',
                    unsafe_allow_html=True,
                )
            with _rb:
                if st.button("✕", key=f"rm_{_qi}", help="Remove"):
                    st.session_state.video_queue.pop(_qi)
                    st.rerun()
        st.button(
            "Clear all", key="clear_queue",
            on_click=lambda: st.session_state.video_queue.clear(),
        )
    else:
        st.markdown(
            '<p class="studio-empty">Add NASA images from chat using the '
            '<strong>📌 Add to queue</strong> button.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── PROMPT ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="studio-section">PROMPT '
        '<span class="studio-optional">(optional)</span></div>',
        unsafe_allow_html=True,
    )
    _video_prompt: str = st.text_area(
        label="prompt_label",
        label_visibility="collapsed",
        placeholder="Describe cinematic style, mood or focus…",
        height=72,
        key="studio_prompt",
    )
    _mode = st.radio(
        "mode",
        ["I2V — use queued images", "T2V — text only"],
        index=0 if _q else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="studio_mode",
    )
    _use_i2v = _mode.startswith("I2V")

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── GENERATE / CANCEL ────────────────────────────────────────────────────
    _running = st.session_state._pipeline_running
    if _running:
        if st.button("✕  Cancel", key="cancel_pipeline", use_container_width=True):
            ev = st.session_state._pipeline_cancel
            if ev:
                ev.set()
    else:
        _can_gen = bool((_video_prompt or "").strip() or _q)
        if st.button(
            "▶  Generate Video",
            type="primary",
            use_container_width=True,
            disabled=not _can_gen,
            key="generate_btn",
        ):
            _prompt = (_video_prompt or "").strip() or (
                " ".join(i.get("caption", "") for i in _q[:3])
            )
            _launch_pipeline(_prompt, _use_i2v)
            st.rerun()

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── PIPELINE STATUS ───────────────────────────────────────────────────────
    st.markdown('<div class="studio-section">PIPELINE</div>', unsafe_allow_html=True)

    # Drain the live update queue on every render cycle
    _live_pq = st.session_state._pipeline_queue
    if _live_pq is not None:
        try:
            while True:
                _item = _live_pq.get_nowait()
                if _item is None:  # sentinel: thread finished
                    st.session_state._pipeline_running = False
                    st.session_state._pipeline_queue = None
                    _running = False
                    break
                st.session_state._pipeline_updates.append(_item)
        except queue.Empty:
            pass

    _updates = st.session_state._pipeline_updates
    _running  = st.session_state._pipeline_running

    if _updates or _running:
        # Visual step row
        _statuses = _pipeline_stage_statuses(_updates)
        st.markdown(_pipeline_steps_html(_statuses), unsafe_allow_html=True)

        # Detailed pipeline execution log
        _log = _pipeline_log_html(_updates)
        if _log:
            st.markdown(_log, unsafe_allow_html=True)

        # Error display
        for upd in _updates:
            if upd.get("stage") == "error":
                _detail = upd.get("detail", "")
                if "AllocationQuota" in _detail or "FreeTier" in _detail:
                    st.error(
                        "**Quota exhausted.** Open the "
                        "[DashScope console](https://dashscope.console.aliyun.com/) "
                        "and disable **\"Use free tier only\"**."
                    )
                else:
                    st.error(f"**Pipeline error:** {_detail}")
                break

        # Poll interval while running
        if _running:
            time.sleep(2)
            st.rerun()

    else:
        st.markdown(
            '<p class="studio-empty">Pipeline status will appear here once '
            "you generate a video.</p>",
            unsafe_allow_html=True,
        )

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    _manifest: dict = {}
    for upd in _updates:
        if upd.get("stage") == "done":
            _manifest = upd.get("manifest", {})

    if _manifest or (not _running and any(u.get("stage") == "video" for u in _updates)):
        st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)
        st.markdown('<div class="studio-section">OUTPUT</div>', unsafe_allow_html=True)

        _clips_dir = _session_dir / "clips"
        _m_clips = [Path(p) for p in _manifest.get("clips", []) if Path(p).exists()]
        _scene_clips = _m_clips or (
            sorted(_clips_dir.glob("scene_*.mp4"), key=lambda p: p.stat().st_mtime)
            if _clips_dir.exists() else []
        )
        for _clip in _scene_clips:
            st.caption(_clip.name)
            st.video(str(_clip))

        # Persist to RunDB once
        if _scene_clips and not st.session_state._pipeline_run_saved:
            _db.save_run(
                run_id=str(uuid.uuid4()),
                conversation_id=st.session_state.conversation_id,
                user_message=st.session_state._pipeline_prompt,
                assistant_response=f"{len(_scene_clips)} clip(s) generated.",
                assets=st.session_state._pipeline_merged_assets,
                manifest=_manifest,
                messages=[
                    Message(role=m["role"], content=m["content"],
                            timestamp=datetime.now().isoformat())
                    for m in st.session_state.messages
                ],
            )
            st.session_state._pipeline_run_saved = True


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDER
# ══════════════════════════════════════════════════════════════════════════════

# On hard browser refresh session_state is cleared but the conversation_id is
# restored from the DB (see session.py). Reload the message history so the
# chat panel isn't blank.
if not st.session_state.messages:
    _init_hist = _db.get_conversation_history(st.session_state.conversation_id)
    if _init_hist:
        st.session_state.messages = _messages_from_history(_init_hist)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:6px 4px 16px; display:flex; align-items:center; gap:10px;">
            <div style="font-size:1.15rem; font-weight:700; color:#ececec;
                        letter-spacing:-0.02em;">WILL.AI</div>
            <div style="font-size:0.68rem; color:rgba(255,255,255,0.3);
                        margin-top:1px;">What Infinity Looks Like</div>
        </div>
        <div style="display:inline-flex; align-items:center; gap:5px;
            background:linear-gradient(90deg,#FF6A00,#EE0979);
            color:#fff; font-size:10px; font-weight:700; letter-spacing:0.04em;
            padding:3px 8px; border-radius:5px; margin-bottom:12px;">
            &#9729; Powered by Alibaba Cloud
        </div>
        <div style="font-size:10px; color:rgba(255,255,255,0.28);
                    margin-bottom:14px; line-height:1.6;">
            LLM &amp; Video: DashScope API<br>
            Models: Qwen 3.7-plus · Wan 2.7 i2v/t2v
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True, key="new_chat_btn", type="primary"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Conversation list grouped by date ──────────────────────────────────
    conversations = _db.list_conversations()
    if conversations:
        today     = date.today()
        yesterday = today - timedelta(days=1)
        week_ago  = today - timedelta(days=7)
        groups: dict[str, list] = {
            "Today": [], "Yesterday": [], "Previous 7 Days": [], "Earlier": [],
        }
        for conv in conversations:
            try:
                conv_date = datetime.fromisoformat(conv["created_at"]).date()
            except Exception:
                groups["Earlier"].append(conv)
                continue
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif conv_date >= week_ago:
                groups["Previous 7 Days"].append(conv)
            else:
                groups["Earlier"].append(conv)

        for group_name, convs in groups.items():
            if not convs:
                continue
            st.markdown(
                f'<span class="conv-group-label">{group_name}</span>',
                unsafe_allow_html=True,
            )
            for conv in convs:
                title = (conv.get("title") or "Untitled")[:36]
                if st.button(title, use_container_width=True,
                              key=f"conv_{conv['conversation_id']}"):
                    st.session_state.conversation_id = conv["conversation_id"]
                    history = _db.get_conversation_history(conv["conversation_id"])
                    st.session_state.messages = _messages_from_history(history)
                    st.rerun()
    else:
        st.caption("No conversations yet.")



# ── Dynamic CSS: expand layout + align input bar when studio is open ──────────
_studio_open = st.session_state.get("studio_open", False)
if _studio_open:
    # CSS-based positioning (sticky/fixed) fails reliably in Streamlit because
    # stAppViewContainer applies a CSS transform for sidebar animations, which
    # breaks position:fixed, and stMain's overflow settings vary across builds,
    # making position:sticky unreliable. The only guaranteed approach is to give
    # the chat its OWN scroll container (st.container(height=X)) so the page
    # itself never needs to scroll — the studio column stays in place naturally.
    st.markdown(
        """<style>
        /* ── Full-width layout, no vertical padding ─────────────────── */
        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {
            max-width: 100% !important;
            padding: 0 0 0 2rem !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }

        /* ── Chat scroll container (st.container(height=X)) ─────────── */
        /* CSS overrides the fixed-px height Streamlit sets so it fills   */
        /* the viewport. border:none removes the default focus ring.      */
        /* padding-right creates breathing room between the content and   */
        /* the container's scrollbar, matching the closed-studio feel.    */
        [data-testid="stVerticalBlockBorderWrapper"] {
            height: calc(100vh - 75px) !important;
            max-height: calc(100vh - 75px) !important;
            border: none !important;
            border-radius: 0 !important;
            background: transparent !important;
            padding-right: 2rem !important;
            box-sizing: border-box !important;
        }

        /* ── Studio column: visual styling only (height set via JS) ─── */
        /* Exclude horizontal blocks inside stChatMessage (image grids). */
        [data-testid="stHorizontalBlock"]:not([data-testid="stChatMessage"] [data-testid="stHorizontalBlock"]) > [data-testid="column"]:last-child:not(:first-child) {
            background: #212121 !important;
            border-left: 1px solid #2a2a2a !important;
            overflow-y: auto !important;
            box-sizing: border-box !important;
        }

        /* ── Bottom bar ─────────────────────────────────────────────── */
        [data-testid="stBottom"] { z-index: 200 !important; }
        /* padding-right keeps input bar aligned with the padded chat    */
        /* content: studio width + same 2rem gap as the scroll container */
        [data-testid="stBottomBlockContainer"] {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: calc((100vw - 258px) * 0.421 + 2rem) !important;
            box-sizing: border-box !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

# ── Chat input (always rendered at page bottom by Streamlit) ──────────────────
user_input = st.chat_input("Ask anything about the universe…")

# Fire a suggestion-click prompt if one is pending
if not user_input and st.session_state.get("pending_prompt"):
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Column layout ─────────────────────────────────────────────────────────────
if _studio_open:
    _col_chat, _col_studio = st.columns([11, 8], gap="medium")
else:
    _col_chat   = st.container()
    _col_studio = None

# When studio is open, the chat messages live inside a fixed-height scroll
# container (_msg_area). This gives the chat its own independent scroll
# context so the page itself never scrolls — the studio column stays in
# place without any CSS positioning tricks. We create _msg_area inside
# _col_chat; Streamlit preserves the DeltaGenerator association after the
# `with` exits, so we render the content in a separate `with _msg_area:`.
with _col_chat:
    if _studio_open:
        _msg_area = st.container(height=1200, border=False)
    else:
        _msg_area = st.container()

# ── Chat messages (inside _msg_area → _col_chat) ──────────────────────────────
with _msg_area:
    if not st.session_state.messages and not user_input:
        # ── Welcome screen ─────────────────────────────────────────────────
        st.markdown(
            """
            <div style="height:42vh; display:flex; flex-direction:column;
                        align-items:center; justify-content:flex-end;
                        text-align:center; gap:0.9rem; padding-bottom:1.5rem;">
                <p style="font-size:2rem; font-weight:700; color:#ececec;
                          margin:0; letter-spacing:-0.03em; line-height:1.1;">
                    What can I help with?
                </p>
                <p style="font-size:0.9rem; color:rgba(255,255,255,0.35);
                          margin:0; max-width:400px; line-height:1.75;">
                    Ask me anything about the universe — a star's life cycle, the
                    latest solar storm, an exoplanet's atmosphere, or what the
                    rover saw on Mars today.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="small")
        for i, (col, prompt) in enumerate(zip([c1, c2, c1, c2], _SUGGESTIONS)):
            with col:
                st.markdown('<div class="suggest-card">', unsafe_allow_html=True)
                if st.button(prompt, use_container_width=True, key=f"sug_{i}"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="chat-footer">WILL.AI · Qwen 3.7 + Live NASA Data</div>',
            unsafe_allow_html=True,
        )

    else:
        # ── Conversation history ────────────────────────────────────────────
        for idx_m, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant":
                    msg_retrieved = message.get("retrieved_passages", [])
                    if msg_retrieved:
                        with st.expander("Retrieved sources", expanded=False):
                            for p in msg_retrieved:
                                snippet = (p.get("snippet") or "").strip()
                                source  = p.get("source") or "source"
                                doc     = p.get("doc_id") or ""
                                if snippet:
                                    st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                                else:
                                    st.markdown(f"**{source}** — {doc}")

                    msg_imgs     = message.get("assets", {}).get("images", [])
                    msg_data_ctx = message.get("assets", {}).get("data", {})
                    msg_desc     = message.get("content", "")
                    if msg_imgs:
                        _img_cols = st.columns(min(len(msg_imgs), 3))
                        for i, (_ic, img) in enumerate(zip(_img_cols, msg_imgs[:3])):
                            with _ic:
                                _t = fetch_thumb(img.get("thumb_url") or img.get("url", ""))
                                st.image(_t, caption=img.get("caption", "")[:50])
                                _already = any(
                                    item["url"] == img.get("url", "")
                                    for item in st.session_state.video_queue
                                )
                                if st.button(
                                    "✓ Queued" if _already else "📌 Add to queue",
                                    key=f"q_hist_{idx_m}_{i}",
                                    disabled=_already,
                                    use_container_width=True,
                                ):
                                    _queue_image(img, msg_desc, msg_data_ctx)
                                    st.session_state.studio_open = True
                                    st.rerun()

        # ── Process new user input ──────────────────────────────────────────
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                qwen_client = QwenClient(api_key=QWEN_API_KEY, model=MODEL_CHAT)
                chat_agent  = ChatAgent(qwen_client)
                history     = _db.get_conversation_history(st.session_state.conversation_id)

                with st.chat_message("assistant"):
                    result: dict = {}

                    _thinking = st.empty()
                    _thinking.markdown(
                        '<span style="color:rgba(255,255,255,0.35); font-size:0.875rem;">'
                        "Connecting to Qwen…</span>",
                        unsafe_allow_html=True,
                    )

                    _gen = chat_agent.answer_stream_internal(user_input, history, result)
                    _seen_first = [False]

                    def _stream():
                        for _chunk in _gen:
                            if not _seen_first[0]:
                                _thinking.empty()
                                _seen_first[0] = True
                            yield _chunk

                    answer = st.write_stream(_stream())
                    answer = answer if isinstance(answer, str) else "".join(answer)
                    if not answer and result.get("answer"):
                        answer = result["answer"]

                    retrieved = result.get("retrieved_passages", [])
                    if retrieved:
                        with st.expander("Retrieved sources", expanded=False):
                            for p in retrieved:
                                snippet = (p.get("snippet") or "").strip()
                                source  = p.get("source") or "source"
                                doc     = p.get("doc_id") or ""
                                if snippet:
                                    st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                                else:
                                    st.markdown(f"**{source}** — {doc}")

                    _turn_assets = result.get("chat_assets", {})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "assets": _turn_assets,
                        "retrieved_passages": retrieved,
                    })
                    _save_chat_turn(user_input, answer, _turn_assets, retrieved)

                # Rerun so images render from history loop with stable keys
                st.rerun()

            except (AuthenticationError, APIError) as exc:
                st.error(f"**Chat error:** {exc}")
            except Exception as exc:
                st.error(f"**Unexpected error:** {exc}")

        st.markdown(
            '<div class="chat-footer">WILL.AI · Qwen 3.7 + Live NASA Data</div>',
            unsafe_allow_html=True,
        )

    # ── JS: set studio column height + auto-scroll chat to bottom ────────────
    # Timestamp in the script content forces Streamlit to reload the iframe
    # on every render, so the studio column is re-styled and the chat
    # container scrolls to the bottom after each new message.
    if _studio_open:
        components.html(
            f"""<script>
(function(){{var _t={time.time():.0f};
var doc=window.parent.document,
    mc=doc.querySelector('[data-testid="stMainBlockContainer"]');
if(!mc)return;
var hbs=mc.querySelectorAll('[data-testid="stHorizontalBlock"]');
for(var i=0;i<hbs.length;i++){{
  var hb=hbs[i],cols=hb.querySelectorAll(':scope>[data-testid="column"]');
  if(hb.closest('[data-testid="stChatMessage"]')||cols.length!==2)continue;
  cols[1].style.height='calc(100vh - 75px)';
  cols[1].style.maxHeight='calc(100vh - 75px)';
  cols[1].style.overflowY='auto';
  var w=cols[0].querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
  if(w)w.scrollTop=w.scrollHeight;
  break;
}}
}})();
</script>""",
            height=0,
        )

# ── Studio panel column ───────────────────────────────────────────────────────
if _col_studio is not None:
    with _col_studio:
        _render_studio_panel()
