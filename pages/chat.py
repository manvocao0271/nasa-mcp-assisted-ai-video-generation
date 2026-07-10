"""Chat page + integrated Video Studio side panel — WILL.ai."""
from __future__ import annotations

import os
import base64
import html as _html
import json
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
    "Show me the latest coronal mass ejections",
    "Fetch recent EPIC Earth images from space",
    "Which near-Earth objects are potentially hazardous?",
    "What are the most recently discovered exoplanets?",
    "Show me a geomagnetic storm from the past month",
    "Find super-Earths with similar mass to our planet",
]

# ── Pipeline stage definitions ─────────────────────────────────────────────────
_PIPELINE_STAGES = [
    ("script",     "", "Script"),
    ("storyboard", "", "Storyboard"),
    ("video",      "", "Video"),
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
    _save_studio_state()


def _save_studio_state() -> None:
    """Persist video queue and pipeline state to the session dir.

    Called after every meaningful change so a hard browser refresh
    restores the Video Studio to exactly the state the user left it in.
    """
    try:
        state = {
            "video_queue":       st.session_state.video_queue,
            "pipeline_updates":  st.session_state._pipeline_updates,
            "pipeline_run_saved": st.session_state._pipeline_run_saved,
            "studio_open":       st.session_state.studio_open,
        }
        (_session_dir / "studio_state.json").write_text(json.dumps(state))
    except Exception:
        pass  # non-critical — never surface a disk error to the user


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
        "script":     ("", "Script"),
        "storyboard": ("", "Storyboard"),
        "video":      ("", "Video"),
        "data":       ("", "Data"),
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
        html.append(f'<div class="log-warning">{warn}</div>')

    html.append("</div>")
    return "".join(html)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO STUDIO PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _launch_pipeline(prompt_text: str) -> None:
    """Build merged_assets and start the background pipeline thread.

    Always uses I2V when images are queued.  If the video model
    falls back to T2V at runtime (e.g. quota exhausted), that is
    reported transparently via the pipeline log warning.
    """
    _q = st.session_state.video_queue
    if _q:
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
    st.markdown('<p class="studio-title">Video Studio</p>', unsafe_allow_html=True)

    # ── INPUTS ───────────────────────────────────────────────────────────────
    st.markdown('<div class="studio-section">REFERENCE IMAGES (max 3)</div>', unsafe_allow_html=True)

    if _q:
        for _qi, _item in enumerate(_q):
            _label = (_item.get("caption") or _item.get("source") or "Image").strip()
            if len(_label) > 58:
                _label = _label[:55] + "…"
            _ra, _rb = st.columns([9, 1])
            with _ra:
                st.markdown(
                    f'<div class="queue-label">{_label}</div>',
                    unsafe_allow_html=True,
                )
            with _rb:
                if st.button("✕", key=f"rm_{_qi}", help="Remove"):
                    st.session_state.video_queue.pop(_qi)
                    _save_studio_state()
                    st.rerun()

    else:
        st.markdown(
            '<p class="studio-empty">Click any NASA image in the chat to add it here.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── PROMPT ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="studio-section">PROMPT (optional, t2v fallback)',
        unsafe_allow_html=True,
    )
    _video_prompt: str = st.text_area(
        label="prompt_label",
        label_visibility="collapsed",
        placeholder="Describe cinematic style, mood or focus…",
        height=72,
        key="studio_prompt",
    )

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── GENERATE / CANCEL ────────────────────────────────────────────────────
    _running = st.session_state._pipeline_running
    if _running:
        if st.button("Cancel", key="cancel_pipeline", use_container_width=True):
            ev = st.session_state._pipeline_cancel
            if ev:
                ev.set()
    else:
        _can_gen = bool((_video_prompt or "").strip() or _q)
        if st.button(
            "Generate Video",
            type="primary",
            use_container_width=True,
            disabled=not _can_gen,
            key="generate_btn",
        ):
            _prompt = (_video_prompt or "").strip() or (
                " ".join(i.get("caption", "") for i in _q[:3])
            )
            _launch_pipeline(_prompt)
            st.rerun()

    st.markdown('<hr class="studio-hr">', unsafe_allow_html=True)

    # ── PIPELINE STATUS ───────────────────────────────────────────────────────
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

    # Persist latest state so a browser refresh restores the pipeline log and clips.
    _save_studio_state()

    if _updates or _running:
        st.markdown('<div class="studio-section">PIPELINE</div>', unsafe_allow_html=True)
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
            st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

        # Persist to RunDB once
        if _scene_clips and not st.session_state._pipeline_run_saved:
            _db.save_run(
                run_id=str(uuid.uuid4()),
                conversation_id=st.session_state.conversation_id,
                user_message=st.session_state._pipeline_prompt,
                assistant_response=f"{len(_scene_clips)} clip(s) generated.",
                assets=st.session_state._pipeline_merged_assets,
                manifest=_manifest,
                messages=[...],
            )
            st.session_state._pipeline_run_saved = True
            _save_studio_state()


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

# Restore Video Studio state (queue, pipeline log, generated clips) on hard
# refresh.  Uses a one-shot flag so the file is only read once per session.
if "_studio_state_restored" not in st.session_state:
    st.session_state._studio_state_restored = True
    _sf = _session_dir / "studio_state.json"
    if _sf.exists():
        try:
            _ss = json.loads(_sf.read_text())
            st.session_state.video_queue        = _ss.get("video_queue", [])
            st.session_state._pipeline_updates  = _ss.get("pipeline_updates", [])
            st.session_state._pipeline_run_saved= _ss.get("pipeline_run_saved", False)
            if _ss.get("studio_open"):
                st.session_state.studio_open = True
        except Exception:
            pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 4px 12px; display:flex; flex-direction:column;
                    align-items:center; gap:4px; text-align:center;">
            <div style="font-size:1.65rem; font-weight:800; color:#ececec;
                        letter-spacing:-0.03em; line-height:1.1;">WILL.ai</div>
            <div style="font-size:0.65rem; color:rgba(255,255,255,0.3);
                        letter-spacing:0.04em;">What Infinity Looks Like</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+  New chat", use_container_width=True, key="new_chat_btn", type="primary"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    _studio_label = "✕  Close Studio" if st.session_state.get("studio_open") else "Video Studio"
    if st.button(_studio_label, use_container_width=True, key="toggle_studio_btn"):
        st.session_state.studio_open = not st.session_state.get("studio_open", False)
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

    # ── Branding footer (bottom of sidebar) ───────────────────────────────
    st.markdown('<div style="flex:1;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="padding:14px 4px 6px; border-top:1px solid #2a2a2a; margin-top:auto;">
            <div style="display:inline-flex; align-items:center; gap:5px;
                background:linear-gradient(90deg,#FF6A00,#EE0979);
                color:#fff; font-size:9px; font-weight:700; letter-spacing:0.04em;
                padding:3px 8px; border-radius:5px; margin-bottom:8px;">
                &#9729; Powered by Alibaba Cloud
            </div>
            <div style="font-size:9px; color:rgba(255,255,255,0.25); line-height:1.7;">
                LLM &amp; Video: DashScope API<br>
                Models: Qwen 3.7-plus · Wan 2.7 i2v/t2v
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ── Always-on CSS: scroll lock + chat container styling ──────────────────────
# The chat always uses its own internal scroll container so the page never
# scrolls. Applied regardless of studio state.
_studio_open = st.session_state.get("studio_open", False)
st.markdown(
    f"""<style>
    /* ── Lock page scroll always ────────────────────────────────────── */
    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"] {{
        overflow: hidden !important;
        max-height: 100vh !important;
        padding-bottom: 0 !important;
    }}
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .block-container {{
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }}
    /* ── Chat border is drawn by JS on chatCol ────────────────────── */
    /* Strip Streamlit's own border on the wrapper so it doesn't double up */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        scrollbar-gutter: stable !important;
        box-sizing: border-box !important;
    }}
    /* ── Bottom bar ──────────────────────────────────────────────────── */
    /* stBottom sits on top of the chat container's bottom border.
       A transparent-to-opaque gradient at the top of stBottom lets the
       border show through the first 16px of the bar's background. */
    [data-testid="stBottom"] {{
        z-index: 200 !important;
        background: linear-gradient(to bottom,
            transparent 0px,
            #000000 16px) !important;
        background-color: transparent !important;
    }}
    [data-testid="stBottomBlockContainer"] {{
        max-width: 100% !important;
        {'padding-left: calc((100vw - 300px) * 0.0878) !important; padding-right: calc((100vw - 300px) * 0.4745) !important;' if _studio_open else 'padding-left: calc((100vw - 300px) * 0.2768) !important; padding-right: calc((100vw - 300px) * 0.2778) !important;'}
        box-sizing: border-box !important;
    }}
    {'/* ── Studio column border wrapper ── */ [data-testid="stHorizontalBlock"]:not([data-testid="stChatMessage"] [data-testid="stHorizontalBlock"]):not([data-testid="stColumn"] [data-testid="stHorizontalBlock"]) > [data-testid="stColumn"]:last-child:not(:first-child) [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; outline: none !important; height: 100% !important; max-height: 100% !important; }' if _studio_open else ''}
    </style>""",
    unsafe_allow_html=True,
)

# ── Chat input (always rendered at page bottom by Streamlit) ──────────────────
user_input = st.chat_input(
    "Ask anything about the universe…",
    disabled=st.session_state._chat_streaming,
)

# Fire a suggestion-click prompt if one is pending
if not user_input and st.session_state.get("pending_prompt"):
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Column layout ─────────────────────────────────────────────────────────────
# Studio open:  [11, 8]     — chat left, studio right
# Studio closed: [4, 11, 4] — chat centred, equal spacers either side
# Always render 3 columns so React never unmounts/remounts the column set
# (which would flash a duplicate layout during the toggle transition).
# Studio open:  tiny left pad + chat + studio  →  [0.1, 11, 8]
# Studio closed: balanced pads + chat          →  [4,   11, 4]
if _studio_open:
    _col_left, _col_chat, _col_studio = st.columns([0.1, 11, 8], gap="medium")
else:
    _col_left, _col_chat, _col_studio = st.columns([4, 11, 4], gap="medium")

# MessageScroller pattern: the scroll viewport (_msg_area / stVerticalBlockBorderWrapper)
# wraps the FULL chat column width so its scrollbar sits at the right edge of the
# chat column (just before the studio panel). The [1,7] indent columns live INSIDE
# the viewport so the left-padding scrolls with the content.
with _col_chat:
    # Always use a height-constrained container so the chat scrolls
    # internally and the page itself never scrolls.
    _msg_area = st.container(border=False)

# Left-indent columns are created inside the scroll viewport; _content_area is
# the render target for all messages.
# Both modes use the same [1, 6, 0.5] inner shim so text layout is identical.
# Centering in non-studio mode comes from the [4, 11, 4] outer columns.
with _msg_area:
    _pad_col, _chat_inner, _rpad_col = st.columns([0.5, 6.5, 0.5], gap=None)
_content_area = _chat_inner

# ── Chat messages (inside _content_area → _msg_area → _col_chat) ──────────────
with _content_area:
    if not st.session_state.messages and not user_input:
        # ── Welcome screen ─────────────────────────────────────────────────
        st.markdown(
            """
            <div style="display:flex; flex-direction:column;
                        align-items:center; justify-content:flex-end;
                        text-align:center; gap:0.65rem; padding:10vh 0 1rem;">
                <p style="font-size:2rem; font-weight:700; color:#ececec;
                          margin:0; letter-spacing:-0.03em; line-height:1.1;">
                    What can I help with?
                </p>
                <p style="font-size:0.82rem; color:rgba(255,255,255,0.38);
                          margin:0; max-width:520px; line-height:1.65;">
                    How it works: ask anything about the universe and WILL.ai will answer with NASA-backed data. Then choose what WILL.ai can make into a video.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Vertical sliding carousel — rendered FIRST so it sits directly under
        # the intro text with no gap. The hidden Streamlit buttons come AFTER in
        # the layout so they don't push the carousel down.
        # Infinite loop: 3 copies, start in middle copy (offset N), snap on transitionend.
        _sug_json = json.dumps(_SUGGESTIONS)
        components.html(
            f"""<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;}}
.wrap{{width:60%;margin:0 auto;}}
.viewport{{
  overflow:hidden;position:relative;
  -webkit-mask-image:linear-gradient(to bottom,transparent 0%,black 18%,black 82%,transparent 100%);
  mask-image:linear-gradient(to bottom,transparent 0%,black 18%,black 82%,transparent 100%);
}}
.track{{display:flex;flex-direction:column;gap:8px;will-change:transform;}}
.track.animating{{transition:transform .32s cubic-bezier(.4,0,.2,1);}}
.item{{
  background:#2a2a2a;border:1px solid #3a3a3a;border-radius:10px;
  color:#d1d1d1;padding:11px 16px;cursor:pointer;text-align:center;
  font-size:14px;line-height:1.45;
  transition:background .15s,border-color .15s,color .15s;
  width:100%;font-family:inherit;flex-shrink:0;display:block;
}}
.item:hover{{background:#333;border-color:#505050;color:#fff;}}
.item:active{{background:#3a3a3a;}}
</style>
<div class="wrap">
  <div class="viewport" id="vp"><div class="track" id="tr"></div></div>
</div>
<script>
(function(){{
  var SUGS={_sug_json};
  var N=SUGS.length;
  var VISIBLE=5;
  var itemH=0;
  var current=N;
  var tr=document.getElementById('tr');
  var vp=document.getElementById('vp');

  for(var copy=0;copy<3;copy++){{
    SUGS.forEach(function(s){{
      var btn=document.createElement('button');
      btn.className='item';
      btn.textContent=s;
      (function(text){{btn.addEventListener('click',function(){{triggerSug(text);}})}})(s);
      tr.appendChild(btn);
    }});
  }}

  function setPos(animate){{
    if(animate){{tr.classList.add('animating');}}
    else{{tr.classList.remove('animating');void tr.offsetHeight;}}
    tr.style.transform='translateY(-'+(current*itemH)+'px)';
  }}

  tr.addEventListener('transitionend',function(){{
    if(current>=2*N){{current-=N;setPos(false);}}
    else if(current<N){{current+=N;setPos(false);}}
  }});

  function measure(){{
    var items=tr.querySelectorAll('.item');
    if(!items.length)return false;
    var h=items[0].offsetHeight;
    if(!h)return false;
    itemH=h+8;
    vp.style.height=(VISIBLE*itemH-8)+'px';
    return true;
  }}

  vp.addEventListener('wheel',function(e){{
    e.preventDefault();
    current+=(e.deltaY>0)?1:-1;
    setPos(true);
  }},{{passive:false}});

  function triggerSug(text){{
    var doc=window.parent.document;
    var wrappers=doc.querySelectorAll('[data-testid="stBaseButton-secondary"]');
    for(var i=0;i<wrappers.length;i++){{
      if((wrappers[i].innerText||'').trim()===text){{
        (wrappers[i].querySelector('button')||wrappers[i]).click();
        return;
      }}
    }}
  }}

  function hideSugBtns(){{
    var doc=window.parent.document;
    doc.querySelectorAll('[data-testid="stBaseButton-secondary"]').forEach(function(w){{
      if(SUGS.indexOf((w.innerText||'').trim())>=0){{
        // Walk up to collapse the full element-container, not just the button wrapper
        var c=w.parentElement;
        while(c&&c.parentElement&&
              c.parentElement.getAttribute('data-testid')==='stVerticalBlock'===false&&
              !c.classList.contains('stElementContainer')&&
              !c.classList.contains('element-container')){{
          c=c.parentElement;
        }}
        if(c){{
          c.style.setProperty('display','none','important');
          c.style.setProperty('height','0','important');
          c.style.setProperty('min-height','0','important');
          c.style.setProperty('margin','0','important');
          c.style.setProperty('padding','0','important');
        }}
      }}
    }});
  }}

  function init(){{
    if(!measure())return false;
    setPos(false);
    hideSugBtns();
    return true;
  }}

  if(!init()){{
    requestAnimationFrame(function(){{
      requestAnimationFrame(function(){{
        if(!init()){{
          setTimeout(function(){{init();}},200);
          setTimeout(function(){{init();}},600);
        }}
      }});
    }});
  }}
}})();
</script>""",
            height=340,
        )

        # Hidden Streamlit buttons — placed AFTER the carousel so they don't
        # create a gap between the intro text and the carousel viewport.
        # The carousel JS finds them by text content and clicks them when selected.
        for i, prompt in enumerate(_SUGGESTIONS):
            if st.button(prompt, key=f"sug_{i}"):
                st.session_state.pending_prompt = prompt
                st.rerun()

    else:
        # ── Drain active chat stream queue ─────────────────────────────────
        _cq = st.session_state._chat_stream_queue
        if _cq is not None:
            _stream_done = False
            _stream_err: str | None = None
            try:
                while True:
                    _ci = _cq.get_nowait()
                    if _ci is None:          # sentinel — thread finished
                        _stream_done = True
                        break
                    elif _ci.get("type") == "chunk":
                        st.session_state._chat_stream_buffer += _ci["text"]
                    elif _ci.get("type") == "done":
                        _r = _ci.get("result", {})
                        st.session_state._chat_stream_retrieved = _r.get("retrieved_passages", [])
                        st.session_state._chat_stream_assets   = _r.get("chat_assets", {})
                    elif _ci.get("type") == "error":
                        _stream_err  = _ci.get("detail", "Unexpected error")
                        _stream_done = True
                        break
            except queue.Empty:
                pass

            if _stream_done:
                st.session_state._chat_streaming     = False
                st.session_state._chat_stream_queue  = None
                if _stream_err:
                    st.error(f"**Chat error:** {_stream_err}")
                else:
                    _ans  = st.session_state._chat_stream_buffer
                    _retr = st.session_state._chat_stream_retrieved
                    _ast  = st.session_state._chat_stream_assets
                    _umsg = st.session_state._chat_stream_user_msg
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": _ans,
                        "assets": _ast,
                        "retrieved_passages": _retr,
                    })
                    _save_chat_turn(_umsg, _ans, _ast, _retr)
                    st.session_state._chat_stream_buffer    = ""
                    st.session_state._chat_stream_retrieved = []
                    st.session_state._chat_stream_assets    = {}
                    st.session_state._chat_stream_user_msg  = ""
                # Rerun so images/sources render from history loop with stable keys
                st.rerun()

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
                        _VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".m4v")
                        _img_cols = st.columns(min(len(msg_imgs), 3))
                        for i, (_ic, img) in enumerate(zip(_img_cols, msg_imgs[:3])):
                            with _ic:
                                _asset_url = img.get("url", "")
                                _is_video = _asset_url.lower().split("?")[0].endswith(_VIDEO_EXTS)
                                _already = any(
                                    item["url"] == _asset_url
                                    for item in st.session_state.video_queue
                                )
                                _cap = _html.escape(img.get("caption", "")[:50])
                                if _is_video:
                                    # NASA APOD sometimes returns a video — display it inline.
                                    # Videos are not added to the image queue (AI pipeline
                                    # needs static reference frames, not a pre-made clip).
                                    st.markdown(
                                        f'<div class="queue-img-cell">'
                                        f'<video src="{_html.escape(_asset_url)}" controls '
                                        f'style="width:100%;height:auto;display:block;'
                                        f'border-radius:8px;">'
                                        f'</video>'
                                        f'<div class="queue-img-caption" style="display:flex;'
                                        f'align-items:center;gap:4px;">'
                                        f'<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" '
                                        f'style="opacity:.5;flex-shrink:0"><path d="M8 5v14l11-7z"/></svg>'
                                        f'{_cap}</div></div>',
                                        unsafe_allow_html=True,
                                    )
                                    # no queue button for videos
                                    continue
                                _t = fetch_thumb(img.get("thumb_url") or _asset_url)
                                if isinstance(_t, bytes):
                                    _src = "data:image/jpeg;base64," + base64.b64encode(_t).decode()
                                else:
                                    _src = str(_t)
                                _cap = _html.escape(img.get("caption", "")[:50])
                                _sel = " selected" if _already else ""
                                _border = "outline:3px solid #10a37f;outline-offset:-3px;" if _already else ""
                                st.markdown(
                                    f'<div class="queue-img-cell{_sel}">'
                                    f'<img src="{_src}" style="width:100%;height:auto;'
                                    f'display:block;border-radius:8px;{_border}">'

                                    f'<div class="queue-img-caption">{_cap}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                if not _already:
                                    if st.button(
                                        " ",
                                        key=f"q_hist_{idx_m}_{i}",
                                        help="queue-trigger",
                                        use_container_width=True,
                                    ):
                                        _queue_image(img, msg_desc, msg_data_ctx)
                                        st.session_state.studio_open = True
                                        st.rerun()

        # ── Wire up clickable images via JS ─────────────────────────────────
        # CSS alone can't reliably overlay a button on an image in Streamlit's
        # DOM.  Instead, a tiny zero-height iframe runs JS in the parent doc:
        # it finds each .img-q-marker element, locates the hidden button in
        # the same stVerticalBlock, and forwards clicks on the image to it.
        components.html(
            """
            <script>
            (function() {
                var pd = window.parent.document;

                function setup() {
                    try {
                        pd.querySelectorAll('.queue-img-cell:not(.selected)').forEach(function(cell) {
                            var vb = cell.closest('[data-testid="stVerticalBlock"]');
                            if (!vb) return;
                            var btn = vb.querySelector('button');
                            if (!btn || btn._willSetup) return;
                            btn._willSetup = true;

                            // Hide the button's direct-child-of-vb wrapper
                            var dc = btn;
                            while (dc.parentElement && dc.parentElement !== vb) dc = dc.parentElement;
                            dc.style.setProperty('display', 'none', 'important');
                            dc.style.setProperty('height', '0', 'important');

                            // Forward image clicks to this button instance
                            var img = cell.querySelector('img');
                            if (img) {
                                img.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    btn.click();
                                });
                            }
                        });
                    } catch(e) { console.warn('WILL.ai img-click:', e); }
                }

                setup();
                setTimeout(setup, 300);
                setTimeout(setup, 1200);

                // Re-run setup synchronously on DOM additions (before browser paint)
                new MutationObserver(function(mutations) {
                    for (var i = 0; i < mutations.length; i++) {
                        if (mutations[i].addedNodes.length > 0) { setup(); break; }
                    }
                }).observe(pd.body, {childList: true, subtree: true});

            })();
            </script>
            """,
            height=0,
        )

        # ── Live streaming assistant turn ───────────────────────────────────
        # While the background LLM thread is running we show accumulated text
        # and rerun every 300 ms.  The script finishes each cycle quickly, so
        # the Video Studio column is interactive between polls.
        if st.session_state._chat_streaming:
            with st.chat_message("assistant"):
                _buf = st.session_state._chat_stream_buffer
                if _buf:
                    st.markdown(_buf)
                else:
                    st.markdown(
                        """
<style>
@keyframes _will-spin{to{transform:rotate(360deg)}}
@keyframes _will-shimmer{0%{background-position:-200% center}to{background-position:200% center}}
.will-marker{display:inline-flex;align-items:center;gap:6px;padding:2px 0;user-select:none}
.will-marker-icon{display:flex;align-items:center;flex-shrink:0;color:rgba(255,255,255,.35)}
.will-marker-icon svg{animation:_will-spin .85s linear infinite}
.will-marker-text{
  font-size:.8rem;letter-spacing:.01em;
  background:linear-gradient(90deg,rgba(255,255,255,.2) 20%,rgba(255,255,255,.6) 50%,rgba(255,255,255,.2) 80%);
  background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  animation:_will-shimmer 1.4s linear infinite;
}
</style>
<div class="will-marker" role="status">
  <span class="will-marker-icon" aria-hidden="true">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2.5"
              stroke-linecap="round" stroke-dasharray="52 20"/>
    </svg>
  </span>
  <span class="will-marker-text">Connecting to Qwen…</span>
</div>""",
                        unsafe_allow_html=True,
                    )
            time.sleep(0.3)
            st.rerun()

        # ── Process new user input ──────────────────────────────────────────
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            history = _db.get_conversation_history(st.session_state.conversation_id)
            st.session_state._chat_stream_user_msg  = user_input
            st.session_state._chat_stream_buffer    = ""
            st.session_state._chat_stream_retrieved = []
            st.session_state._chat_stream_assets    = {}

            _cq_new: queue.Queue = queue.Queue()
            st.session_state._chat_stream_queue = _cq_new
            st.session_state._chat_streaming    = True

            def _chat_bg(
                _msg=user_input, _hist=history, _q=_cq_new
            ) -> None:
                _result: dict = {}
                try:
                    _qc = QwenClient(api_key=QWEN_API_KEY, model=MODEL_CHAT)
                    _ca = ChatAgent(_qc)
                    for _chunk in _ca.answer_stream_internal(_msg, _hist, _result):
                        _q.put({"type": "chunk", "text": _chunk})
                    _q.put({"type": "done", "result": _result})
                except Exception as _exc:
                    _q.put({"type": "error", "detail": str(_exc)})
                finally:
                    _q.put(None)  # sentinel

            threading.Thread(target=_chat_bg, daemon=True).start()
            st.rerun()

    # ── JS: clamp column heights + scroll chat to bottom (runs every render) ──
    # Handles both [11,8] (studio-open, 2 cols) and [4,11,4] (studio-closed, 3 cols).
    # chat column index: 0 for 2-col layout, 1 for 3-col layout.
    components.html(
        f"""<script>
(function(){{var _t={time.time():.0f};var _hasMessages={'true' if st.session_state.get('messages') else 'false'};var _studioOpen={'true' if _studio_open else 'false'};
var doc=window.parent.document;

// ── 1. Persist scroll lock in <head> ─────────────────────────────────────────
var _ss=doc.getElementById('_will_scroll_lock');
if(!_ss){{_ss=doc.createElement('style');_ss.id='_will_scroll_lock';doc.head.appendChild(_ss);}}
_ss.textContent='html,body{{background:#000!important;}}html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"]{{overflow:hidden!important;max-height:100vh!important;padding-bottom:0!important;}}';

// ── 2. Reset scroll on containers ────────────────────────────────────────────
['stApp','stAppViewContainer','stMainBlockContainer'].forEach(function(id){{
  var el=doc.querySelector('[data-testid="'+id+'"]');
  if(el)el.scrollTop=0;
}});

// ── 3. Find outer horizontal block and clamp columns ─────────────────────────
var mc=doc.querySelector('[data-testid="stMainBlockContainer"]');
if(!mc)return;
mc.style.setProperty('padding-right','2rem','important');
var hbs=mc.querySelectorAll('[data-testid="stHorizontalBlock"]');
for(var i=0;i<hbs.length;i++){{
  var hb=hbs[i],cols=hb.querySelectorAll(':scope>[data-testid="stColumn"]');
  if(hb.closest('[data-testid="stChatMessage"]')||hb.closest('[data-testid="stColumn"]'))continue;
  // Always 3 cols: [left, chat, studio] — chat is always cols[1]
  if(cols.length!==3)continue;
  var chatCol=cols[1];
  hb.style.setProperty('overflow','visible','important');
  cols[2].style.setProperty('overflow-y','auto','important');

  // Measure the gap precisely so the border is always above stBottom
  var stBottomEl=doc.querySelector('[data-testid="stBottom"]');
  var stBottomTop=stBottomEl
    ?stBottomEl.getBoundingClientRect().top
    :window.parent.innerHeight;
  var chatColTop=chatCol.getBoundingClientRect().top;
  var chatH=Math.max(200,stBottomTop-chatColTop-36); // 36px gap so border-radius is fully visible

  // Draw the border ON chatCol itself — JS inline !important beats Emotion
  chatCol.style.setProperty('height',chatH+'px','important');
  chatCol.style.setProperty('max-height',chatH+'px','important');
  chatCol.style.setProperty('overflow','hidden','important');
  chatCol.style.setProperty('padding','0','important');
  chatCol.style.setProperty('background','#212121','important');
  chatCol.style.setProperty('outline','none','important');
  chatCol.style.setProperty('box-shadow','inset 0 0 0 1px rgba(255,255,255,0.35)','important');
  chatCol.style.setProperty('border-radius','12px','important');
  chatCol.style.setProperty('box-sizing','border-box','important');

  // Also size and style the studio column to match (only when open)
  if(_studioOpen){{
    cols[2].style.setProperty('height',chatH+'px','important');
    cols[2].style.setProperty('max-height',chatH+'px','important');
    cols[2].style.setProperty('background','#212121','important');
    cols[2].style.setProperty('border-radius','12px','important');
    cols[2].style.setProperty('box-shadow','inset 0 0 0 1px rgba(255,255,255,0.35)','important');
    cols[2].style.setProperty('box-sizing','border-box','important');
    cols[2].style.setProperty('padding','1rem 0.75rem 2rem','important');
    cols[2].style.removeProperty('width');
    cols[2].classList.add('will-studio-col');
    cols[2].querySelectorAll('[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"]').forEach(function(el){{
      el.style.setProperty('width','100%','important');
      el.style.setProperty('min-width','0','important');
      el.style.setProperty('box-sizing','border-box','important');
    }});
  }} else {{
    // Studio closed — make right spacer col invisible
    cols[2].style.setProperty('background','transparent','important');
    cols[2].style.setProperty('box-shadow','none','important');
    cols[2].style.setProperty('border','none','important');
    cols[2].style.setProperty('outline','none','important');
    cols[2].style.removeProperty('height');
    cols[2].style.removeProperty('max-height');
    cols[2].classList.remove('will-studio-col');
  }}

  // Strip the inner wrapper's own border so we don't get a double border
  var wrapper=chatCol.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
  if(wrapper){{
    wrapper.style.setProperty('border','none','important');
    wrapper.style.setProperty('outline','none','important');
    wrapper.style.setProperty('height','100%','important');
    wrapper.style.setProperty('max-height','100%','important');
    wrapper.style.setProperty('margin','0','important');
    wrapper.style.setProperty('padding','0','important');
  }}
  // Also strip studio col inner wrapper border
  var studioWrapper=cols[2].querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
  if(studioWrapper){{
    studioWrapper.style.setProperty('border','none','important');
    studioWrapper.style.setProperty('outline','none','important');
  }}

  // Scroll chat to the latest visible message every time the view updates.
  var wrapperEl = chatCol.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
  var scrollEl = wrapperEl || chatCol.querySelector('[data-testid="stVerticalBlock"]');
  if(scrollEl) {{
    scrollEl.style.setProperty('overflow-y', _hasMessages ? 'auto' : 'hidden', 'important');
    scrollEl.style.setProperty('overflow-x', 'hidden', 'important');
    scrollEl.style.setProperty('min-height', '0', 'important');
    scrollEl.style.setProperty('height', '100%', 'important');
    scrollEl.style.setProperty('max-height', '100%', 'important');
  }}
  if(_hasMessages && scrollEl){{
    window.parent.requestAnimationFrame(function(){{
      window.parent.requestAnimationFrame(function(){{
        var msgEls = chatCol.querySelectorAll('[data-testid="stChatMessage"]');
        if(msgEls.length > 0){{
          var lastMsg = msgEls[msgEls.length - 1];
          if(lastMsg && typeof lastMsg.scrollIntoView === 'function'){{
            lastMsg.scrollIntoView({{block:'end', inline:'nearest'}});
          }}
        }}
        if(scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
      }});
    }});
  }}
  break;
}}
}})();
</script>""",
        height=0,
    )

components.html(
    """
    <script>
    (function() {
      var doc = window.parent.document;

      function findScrollEl() {
        var mc = doc.querySelector('[data-testid="stMainBlockContainer"]');
        if (!mc) return null;
        var hbs = mc.querySelectorAll('[data-testid="stHorizontalBlock"]');
        for (var i = 0; i < hbs.length; i++) {
          var hb = hbs[i];
          if (hb.closest('[data-testid="stChatMessage"]') || hb.closest('[data-testid="stColumn"]')) continue;
          var cols = hb.querySelectorAll(':scope>[data-testid="stColumn"]');
          if (cols.length !== 3) continue;
          var chatCol = cols[1];
          var wrapper = chatCol.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
          var scrollEl = wrapper || chatCol.querySelector('[data-testid="stVerticalBlock"]');
          if (scrollEl) return scrollEl;
        }
        return null;
      }

      function scrollBottom(el) {
        if (!el) return;
        el.scrollTop = el.scrollHeight;
      }

      function setupObserver() {
        var el = findScrollEl();
        if (!el) {
          setTimeout(setupObserver, 200);
          return;
        }
        scrollBottom(el);
        var observer = new MutationObserver(function() {
          scrollBottom(el);
        });
        observer.observe(el, { childList: true, subtree: true, characterData: true });
      }

      setupObserver();
    })();
    </script>
    """,
    height=0,
)

# ── Studio panel column ───────────────────────────────────────────────────────
with _col_studio:
    if _studio_open:
        _render_studio_panel()
