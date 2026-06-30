"""WILL.AI — What Infinity Looks Like AI

Streamlit two-panel workbench:
  Left  — multi-turn chat with WILL.AI (ChatAgent + live NASA data)
  Right — Video Studio: queue NASA images from chat, generate Wan 2.7 clips

Launch:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import AuthenticationError, APIError

from agent.orchestrator import Orchestrator
from agent.chat_agent import ChatAgent
from agent.qwen_client import QwenClient, MODEL_CHAT
from agent.run_db import RunDB, Message

load_dotenv()

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
DEBUG = 1

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WILL.AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Chrome removal ────────────────────────────────────────────────────────── */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ── Viewport lock: page frame never scrolls ─────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    overflow: hidden !important;
    height: 100vh !important;
}

/* ── Block container: flush to viewport ───────────────────────────────── */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Main three-column row ─────────────────────────────────────────────── */
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    min-height: 100vh !important;
    align-items: stretch !important;
}

/* ── Left panel — conversations ───────────────────────────────────────── */
[data-testid="stColumn"]:has(#convs-panel-anchor) > [data-testid="stVerticalBlock"] {
    padding: 2rem 1.25rem 1rem 1.75rem !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-height: 100vh !important;
    background: rgba(0,0,0,0.08) !important;
}

/* ── Center panel — chat ──────────────────────────────────────────────── */
[data-testid="stColumn"]:has(#chat-panel-anchor) > [data-testid="stVerticalBlock"] {
    padding: 0 4rem 0 4rem !important;
    min-height: 100vh !important;
}

/* ── Right panel — video studio ──────────────────────────────────────── */
[data-testid="stColumn"]:has(#studio-panel-anchor) > [data-testid="stVerticalBlock"] {
    padding: 2rem 1.75rem 1rem 1.25rem !important;
    border-left: 1px solid rgba(255,255,255,0.07) !important;
    min-height: 100vh !important;
    background: rgba(0,0,0,0.08) !important;
}

/* ── Panel section label ─────────────────────────────────────────────────── */
.wil-label {
    display: block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.2);
    margin: 0 0 1.25rem;
    line-height: 1;
}

/* ── Conversations: list buttons ────────────────────────────────────────── */
[data-testid="stColumn"]:has(#convs-panel-anchor) [data-testid="stBaseButton-secondary"] {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 0.77rem !important;
    line-height: 1.4 !important;
    border: none !important;
    background: transparent !important;
    color: rgba(255,255,255,0.45) !important;
    padding: 0.32rem 0.6rem !important;
    border-radius: 4px !important;
    height: auto !important;
    min-height: unset !important;
    transition: background 0.12s, color 0.12s !important;
}
[data-testid="stColumn"]:has(#convs-panel-anchor) [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.9) !important;
}

/* ── Conversations: scroll box ──────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"]:has(#convs-scroll-inner) {
    height: calc(100vh - 8rem) !important;
    max-height: none !important;
    border: none !important;
    border-radius: 0 !important;
}

/* ── Chat: messages scroll box ─────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"]:has(#chat-scroll-inner) {
    height: calc(100vh - 8rem) !important;
    max-height: none !important;
    border: none !important;
}

/* ── Chat messages — base ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    padding: 0.9rem 0 !important;
    gap: 0.9rem !important;
    align-items: flex-start !important;
    border-bottom: none !important;
}

/* ── User avatar: hidden ────────────────────────────────────────────────── */
[data-testid="stChatMessageAvatarUser"] { display: none !important; }

/* ── User message row: block + text-align right (bypasses flex issues) ───── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display: block !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-align: right !important;
}

/* ── User bubble: inline-block so text-align:right places it at the edge ─── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
    display: inline-flex !important;
    align-items: center !important;      /* vertically center text in the bubble */
    vertical-align: middle !important;
    text-align: left !important;         /* reset text inside bubble */
    background: rgba(255,255,255,0.09) !important;
    border-radius: 20px 20px 5px 20px !important;
    padding: 0.72rem 1.1rem !important;
    max-width: 70% !important;
}
/* ── Strip inner-wrapper margins that break vertical centering ───────────── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] p {
    margin: 0 !important;
}
/* ── Assistant avatar: compact ─────────────────────────────────────────── */
[data-testid="stChatMessageAvatarAssistant"] {
    width: 30px !important;
    height: 30px !important;
    min-width: 30px !important;
    margin-top: 3px !important;
}

/* ── Message typography ────────────────────────────────────────────────── */
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    font-size: 1rem !important;
    line-height: 2 !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0 !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.3rem !important;
    line-height: 1.7 !important;
}

/* ── Chat input: rounded, minimal ─────────────────────────────────────── */
[data-testid="stChatInputContainer"] { padding: 0.75rem 0 1rem !important; }
[data-testid="stChatInputContainer"] > div { max-width: 100% !important; margin: 0 !important; }
[data-testid="stChatInputContainer"] [data-baseweb="textarea"] {
    border-radius: 14px !important;
    border-color: rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.04) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatInputContainer"] [data-baseweb="textarea"]:focus-within {
    border-color: rgba(255,255,255,0.28) !important;
    background: rgba(255,255,255,0.06) !important;
}

/* ── Container borders ────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(255,255,255,0.06) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "run_db" not in st.session_state:
    st.session_state.run_db = RunDB()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

for key, default in [
    ("messages", []),
    ("video_queue", []),        # list[{url, thumb_url, caption, source, data_context, description}]
    ("_pipeline_running", False),
    ("_pipeline_updates", []),  # accumulated update dicts from the pipeline generator
    ("_pipeline_queue", None),  # queue.Queue while a run is in flight
    ("_pipeline_cancel", None), # threading.Event — set() to cancel a running pipeline
    ("_pipeline_prompt", ""),
    ("_pipeline_chat_desc", ""),
    ("_pipeline_merged_assets", {}),
    ("_pipeline_run_saved", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_thumb(url: str) -> bytes | str:
    """Return image bytes for domains that block browser access, else the original URL.

    NASA Image Library assets (images-assets.nasa.gov) return 403/timeout when loaded directly by the browser. We fetch them server-side and hand Streamlit raw bytes instead, which always works.
    We also swap ~large.jpg / ~orig.jpg for ~thumb.jpg to keep previews fast.
    """
    _PROXY_DOMAINS = ("images-assets.nasa.gov",)
    if not any(d in url for d in _PROXY_DOMAINS):
        return url  # fast path — most URLs are fine
    # Use the small thumbnail variant for display speed
    thumb_url = url.replace("~large.", "~thumb.").replace("~orig.", "~thumb.").replace("~medium.", "~thumb.")
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            r = client.get(thumb_url)
        if r.status_code < 400:
            return r.content
        # thumb variant failed — try the original URL
        r = client.get(url)
        if r.status_code < 400:
            return r.content
    except Exception:
        pass
    return url  # fall back to URL on failure; Streamlit will show broken-image icon


def _merge_data_contexts(queue: list[dict]) -> dict:
    """Merge data_context dicts from all queued items; later items win on key collision."""
    merged: dict = {}
    for item in queue:
        merged.update(item.get("data_context") or {})
    return merged


def _queue_image(img: dict, description: str, data_context: dict) -> None:
    """Add an image to the video queue, deduplicating by URL."""
    url = img.get("url", "")
    if not url:
        return
    existing_urls = {item["url"] for item in st.session_state.video_queue}
    if url in existing_urls:
        return
    st.session_state.video_queue.append({
        "url": url,
        "thumb_url": img.get("thumb_url", ""),
        "caption": img.get("caption", ""),
        "source": img.get("source", "NASA"),
        "data_context": data_context,
        "description": description,
    })


# ── Pipeline helpers ─────────────────────────────────────────────────────────

ICONS = {
    "data": "—", "script": "—", "storyboard": "—", "video": "—",
}
LABELS = {
    "data":       "Fetching NASA data",
    "script":     "Writing scene captions",
    "storyboard": "Generating storyboard",
    "video":      "Generating video clips (Wan 2.7)",
}


def _render_pipeline_status(updates: list[dict], running: bool) -> dict:
    """Render accumulated pipeline updates inside a st.status block. Returns manifest dict."""
    manifest: dict = {}
    last_per_stage: dict[str, dict] = {}  # final state per pipeline stage
    warnings_list: list[str] = []
    error_detail: str = ""

    for upd in updates:
        stage = upd.get("stage", "")
        if stage == "done":
            manifest = upd.get("manifest", {})
        elif stage == "warning":
            warnings_list.append(upd.get("detail", ""))
        elif stage == "error":
            error_detail = upd.get("detail", "")
        else:
            last_per_stage[stage] = upd

    has_error = bool(error_detail)
    if has_error:
        overall_state, overall_label = "error", "Pipeline error"
    elif not running:
        overall_state, overall_label = "complete", "Pipeline complete."
    else:
        running_stages = [s for s, u in last_per_stage.items() if u.get("status") == "running"]
        lbl = LABELS.get(running_stages[-1], "Running pipeline") if running_stages else "Running pipeline"
        overall_state, overall_label = "running", f"{lbl}…"

    with st.status(overall_label, state=overall_state, expanded=True):
        for stage, upd in last_per_stage.items():
            label = LABELS.get(stage, stage.title())
            if upd.get("status") == "running":
                st.markdown(f"**{label}** &nbsp; *running…*")
            else:
                st.markdown(f"**{label}** &nbsp; {upd.get('detail', '')}")
        for w in warnings_list:
            st.warning(w)
        if has_error:
            if "AllocationQuota" in error_detail or "FreeTier" in error_detail:
                st.error(
                    "**Video generation quota exhausted.** "
                    "Open the [DashScope console](https://dashscope.console.aliyun.com/) "
                    "and disable **\"Use free tier only\"**."
                )
            else:
                st.error(f"**Pipeline error:** {error_detail}")

    return manifest


def _save_chat_turn(user_message: str, assistant_response: str) -> None:
    """Persist a chat turn to RunDB."""
    conv_id = st.session_state.conversation_id
    history_before = st.session_state.run_db.get_conversation_history(conv_id)
    is_first = len(history_before) == 0
    run_id = str(uuid.uuid4())
    st.session_state.run_db.save_run(
        run_id=run_id,
        conversation_id=conv_id,
        user_message=user_message,
        assistant_response=assistant_response,
        assets={},
        manifest={},
        messages=[
            Message(role=m["role"], content=m["content"], timestamp=datetime.now().isoformat())
            for m in st.session_state.messages
        ],
    )
    if is_first:
        st.session_state.run_db.set_conversation_title(conv_id, user_message[:120])


# ── Sidebar — debug only (native Streamlit sidebar hidden from view) ──────────

if DEBUG:
    with st.sidebar.expander("Debug", expanded=False):
        st.write("video_queue:", len(st.session_state.get("video_queue", [])))
        st.write("messages:", len(st.session_state.get("messages", [])))
        st.write("conversation_id:", st.session_state.get("conversation_id", "")[:8])

# ── Guards ────────────────────────────────────────────────────────────────────

if not QWEN_API_KEY:
    st.error("QWEN_API_KEY is not set. Add it to your .env file.")
    st.stop()

# ── Layout: three fixed panels ───────────────────────────────────────────────

col_convs, col_chat, col_studio = st.columns([4, 12, 4], gap="small")

# ── Conversations ─────────────────────────────────────────────────────────────

with col_convs:
    st.markdown('<span id="convs-panel-anchor"></span>', unsafe_allow_html=True)
    st.markdown('<span class="wil-label">Conversations</span>', unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True, key="new_chat_btn"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<hr style="border:none; border-top:1px solid rgba(255,255,255,0.06); margin:0.75rem 0;">',
        unsafe_allow_html=True,
    )

    conversations = st.session_state.run_db.list_conversations()
    if conversations:
        with st.container(height=800, border=False):
            st.markdown('<span id="convs-scroll-inner"></span>', unsafe_allow_html=True)
            for conv in conversations:
                conv_title = conv.get("title") or "Untitled"
                conv_date = conv["created_at"][:10] if conv["created_at"] else ""
                _label = f"{conv_title[:28]}  ·  {conv_date}"
                if st.button(_label, use_container_width=True, key=f"btn_conv_{conv['conversation_id']}"):
                    st.session_state.conversation_id = conv["conversation_id"]
                    history = st.session_state.run_db.get_conversation_history(conv["conversation_id"])
                    st.session_state.messages = []
                    for run in history:
                        st.session_state.messages.append({"role": "user", "content": run["user_message"]})
                        st.session_state.messages.append({"role": "assistant", "content": run["assistant_response"]})
                    st.rerun()
    else:
        st.caption("No conversations yet.")

# ── Video Studio ──────────────────────────────────────────────────────────────

with col_studio:
    st.markdown('<span id="studio-panel-anchor"></span>', unsafe_allow_html=True)
    st.markdown('<span class="wil-label">Video Studio</span>', unsafe_allow_html=True)

    _queue = st.session_state.video_queue

    if _queue:
        st.caption(f"{len(_queue)} image{'s' if len(_queue) != 1 else ''} queued")
        with st.container(height=180, border=True):
            for _qi, _item in enumerate(_queue):
                _c_info, _c_btn = st.columns([5, 1])
                with _c_info:
                    st.caption((_item.get("caption") or _item.get("source") or "NASA image")[:60])
                with _c_btn:
                    if st.button("✕", key=f"rm_{_qi}", help="Remove"):
                        st.session_state.video_queue.pop(_qi)
                        st.rerun()
        if st.button("Clear queue", use_container_width=True):
            st.session_state.video_queue = []
            st.rerun()
    else:
        st.caption("Queue NASA images from chat to generate a video.")

    if _queue and not (st.session_state.get("studio_prompt") or "").strip():
        _last = _queue[-1]
        _auto = (_last.get("description") or _last.get("caption") or "").strip()
        if _auto:
            st.session_state["studio_prompt"] = _auto[:300]

    _video_prompt = st.text_area(
        "Video prompt",
        placeholder="Describe the video you want to generate…",
        height=100,
        key="studio_prompt",
    )
    _mode = st.radio(
        "Mode",
        ["I2V — use queued images", "T2V — text only"],
        index=0 if _queue else 1,
        horizontal=True,
        help="I2V uses queued NASA images as reference frames. T2V fetches NASA data automatically.",
    )
    _use_i2v = _mode.startswith("I2V")

    _generate_clicked = st.button(
        "Generate Video",
        type="primary",
        use_container_width=True,
        disabled=not (_video_prompt or "").strip() or st.session_state._pipeline_running,
    )

    if _generate_clicked and (_video_prompt or "").strip() and not st.session_state._pipeline_running:
        _prompt_text = _video_prompt.strip()
        if _use_i2v and _queue:
            _merged_assets: dict = {
                "query": _prompt_text,
                "images": [
                    {"url": i["url"], "thumb_url": i["thumb_url"],
                     "caption": i["caption"], "source": i["source"]}
                    for i in _queue
                ],
                "data": _merge_data_contexts(_queue),
                "tools_called": [],
            }
            _chat_desc = " ".join(i.get("description", "") for i in _queue)[:400]
        else:
            _orch_fetch = Orchestrator(qwen_api_key=QWEN_API_KEY, nasa_api_key=NASA_API_KEY)
            _merged_assets = {}
            _fetch_status = st.status("Fetching NASA data…", expanded=True)
            try:
                for _update in _orch_fetch.fetch_data(_prompt_text):
                    _merged_assets = _update.get("assets", _merged_assets)
                    if _update["status"] == "running":
                        with _fetch_status:
                            st.markdown(_update["detail"])
                    elif _update["status"] == "done":
                        _fetch_status.update(label=_update["detail"], state="complete")
            except Exception as _exc:
                _fetch_status.update(label="Data fetch failed", state="error")
                st.error(f"**Error fetching NASA data:** {_exc}")
                st.stop()
            _chat_desc = ""

        # Persist context so the post-pipeline render (a later re-run) can use it
        st.session_state._pipeline_prompt = _prompt_text
        st.session_state._pipeline_chat_desc = _chat_desc
        st.session_state._pipeline_merged_assets = _merged_assets
        st.session_state._pipeline_run_saved = False

        # Launch the pipeline in a background thread — never blocks the Streamlit script
        _cancel_ev = threading.Event()
        _q: queue.Queue = queue.Queue()
        st.session_state._pipeline_cancel = _cancel_ev
        st.session_state._pipeline_queue = _q
        st.session_state._pipeline_running = True
        st.session_state._pipeline_updates = []

        def _bg_pipeline(
            _p=_prompt_text, _a=_merged_assets, _d=_chat_desc,
            _q=_q, _ev=_cancel_ev,
        ) -> None:
            _orch = Orchestrator(
                qwen_api_key=QWEN_API_KEY,
                nasa_api_key=NASA_API_KEY,
                cancel_event=_ev,
            )
            try:
                for upd in _orch.run_pipeline(_p, _a, chat_description=_d):
                    _q.put(upd)
            except InterruptedError:
                _q.put({"stage": "warning", "status": "warning",
                        "detail": "Generation cancelled by user."})
            except Exception as exc:
                detail = str(exc)
                if "AllocationQuota" in detail or "FreeTier" in detail:
                    detail = (
                        "Video generation quota exhausted. "
                        "Disable \"Use free tier only\" in the DashScope console."
                    )
                _q.put({"stage": "error", "status": "error", "detail": detail})
            finally:
                _q.put(None)  # sentinel — signals thread completion

        threading.Thread(target=_bg_pipeline, daemon=True).start()
        st.rerun()

    # ── Drain pipeline queue (runs on every re-render while pipeline is in flight) ──
    _pq = st.session_state._pipeline_queue
    if _pq is not None:
        try:
            while True:
                _item = _pq.get_nowait()
                if _item is None:  # sentinel: pipeline thread finished
                    st.session_state._pipeline_running = False
                    st.session_state._pipeline_queue = None
                    break
                st.session_state._pipeline_updates.append(_item)
        except queue.Empty:
            pass

    if st.session_state._pipeline_updates or st.session_state._pipeline_running:
        _manifest = _render_pipeline_status(
            st.session_state._pipeline_updates,
            st.session_state._pipeline_running,
        )

        if st.session_state._pipeline_running:
            if st.button("✕  Cancel", key="cancel_pipeline"):
                _ev = st.session_state._pipeline_cancel
                if _ev:
                    _ev.set()
            time.sleep(2)  # brief poll interval — keeps UI refreshing every 2 s
            st.rerun()

        elif _manifest:
            _manifest_clips = [Path(p) for p in _manifest.get("clips", []) if Path(p).exists()]
            _clips_dir = Path("output/clips")
            _scene_clips = _manifest_clips or (
                sorted(_clips_dir.glob("scene_*.mp4"), key=lambda p: p.stat().st_mtime)
                if _clips_dir.exists() else []
            )
            for _clip in _scene_clips:
                st.caption(_clip.name)
                st.video(str(_clip))

            if not st.session_state._pipeline_run_saved:
                _run_id = str(uuid.uuid4())
                _saved_prompt = st.session_state._pipeline_prompt
                _saved_assets = st.session_state._pipeline_merged_assets
                st.session_state.run_db.save_run(
                    run_id=_run_id,
                    conversation_id=st.session_state.conversation_id,
                    user_message=_saved_prompt,
                    assistant_response=(
                        f"{len(_scene_clips)} clip(s) generated."
                        if _scene_clips else "Pipeline complete."
                    ),
                    assets=_saved_assets,
                    manifest=_manifest,
                    messages=[
                        Message(role=m["role"], content=m["content"], timestamp=datetime.now().isoformat())
                        for m in st.session_state.messages
                    ],
                )
                st.session_state._pipeline_run_saved = True

# ── Chat ────────────────────────────────────────────────────────────────────────

with col_chat:
    st.markdown('<span id="chat-panel-anchor"></span>', unsafe_allow_html=True)

    # Container defined first — renders ABOVE the input bar in the DOM
    msgs = st.container(height=900, border=False)

    # Input defined after — Streamlit places it below msgs (sticky-bottom feel)
    user_input = st.chat_input("Ask anything about the universe…")

    # Fill msgs: everything appears inside the scroll area, above the input
    with msgs:
        st.markdown('<span id="chat-scroll-inner"></span>', unsafe_allow_html=True)

        if not st.session_state.messages and not user_input:
            st.markdown(
                """<div style="height:70vh; display:flex; flex-direction:column;
                            align-items:center; justify-content:center;
                            text-align:center; gap:1.25rem;">
                    <p style="font-size:1.75rem; font-weight:700; color:rgba(255,255,255,0.88);
                              margin:0; letter-spacing:-0.025em; line-height:1.15;">
                        What Infinity Looks Like.
                    </p>
                    <p style="font-size:0.9rem; color:rgba(255,255,255,0.33); line-height:1.8;
                              margin:0; max-width:420px;">
                        Ask me anything about the universe &mdash; a star&rsquo;s life cycle,
                        the latest solar storm, an exoplanet&rsquo;s atmosphere, or what the rover
                        saw on Mars today. I draw on live NASA data to ground every answer.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            for idx_m, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        msg_retrieved = message.get("retrieved_passages", [])
                        if msg_retrieved:
                            with st.expander("Retrieved sources", expanded=False):
                                for p in msg_retrieved:
                                    snippet = (p.get("snippet") or "").strip()
                                    source = p.get("source") or "source"
                                    doc = p.get("doc_id") or ""
                                    if snippet:
                                        st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                                    else:
                                        st.markdown(f"**{source}** — {doc}")

                        msg_imgs = message.get("assets", {}).get("images", [])
                        msg_data_ctx = message.get("assets", {}).get("data", {})
                        msg_desc = message.get("content", "")
                        if msg_imgs:
                            _cols = st.columns(min(len(msg_imgs), 3))
                            for i, (_col, img) in enumerate(zip(_cols, msg_imgs[:3])):
                                with _col:
                                    _t = _fetch_thumb(img.get("thumb_url") or img.get("url", ""))
                                    st.image(_t, caption=img.get("caption", "")[:50], width="stretch")
                                    _queued_urls = {item["url"] for item in st.session_state.video_queue}
                                    _already = img.get("url", "") in _queued_urls
                                    _btn_label = "Queued" if _already else "Add to queue"
                                    if st.button(_btn_label, key=f"q_hist_{idx_m}_{i}", disabled=_already, use_container_width=True):
                                        _queue_image(img, msg_desc, msg_data_ctx)
                                        st.rerun()

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                try:
                    qwen_client = QwenClient(api_key=QWEN_API_KEY, model=MODEL_CHAT)
                    chat_agent = ChatAgent(qwen_client)
                    history = st.session_state.run_db.get_conversation_history(st.session_state.conversation_id)

                    with st.chat_message("assistant"):
                        result: dict = {}
                        answer = st.write_stream(
                            chat_agent.answer_stream_internal(user_input, history, result)
                        )
                        answer = answer if isinstance(answer, str) else "".join(answer)
                        if not answer and result.get("answer"):
                            answer = result["answer"]

                        retrieved = result.get("retrieved_passages", [])
                        if retrieved:
                            with st.expander("Retrieved sources", expanded=False):
                                for p in retrieved:
                                    snippet = (p.get("snippet") or "").strip()
                                    source = p.get("source") or "source"
                                    doc = p.get("doc_id") or ""
                                    if snippet:
                                        st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                                    else:
                                        st.markdown(f"**{source}** — {doc}")

                        _turn_assets = result.get("chat_assets", {})
                        _turn_imgs = _turn_assets.get("images", [])
                        _turn_data_ctx = _turn_assets.get("data", {})

                        if _turn_imgs:
                            _cols = st.columns(min(len(_turn_imgs), 3))
                            for i, (_col, img) in enumerate(zip(_cols, _turn_imgs[:3])):
                                with _col:
                                    _t = _fetch_thumb(img.get("thumb_url") or img.get("url", ""))
                                    st.image(_t, caption=img.get("caption", "")[:50], width="stretch")
                                    _queued_urls = {item["url"] for item in st.session_state.video_queue}
                                    _already = img.get("url", "") in _queued_urls
                                    _btn_label = "Queued" if _already else "Add to queue"
                                    if st.button(_btn_label, key=f"q_live_{i}", disabled=_already, use_container_width=True):
                                        _queue_image(img, answer, _turn_data_ctx)
                                        st.rerun()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "assets": _turn_assets,
                            "retrieved_passages": retrieved,
                        })
                        _save_chat_turn(user_input, answer)

                except (AuthenticationError, APIError) as exc:
                    st.error(f"**Chat error:** {exc}")
                except Exception as exc:
                    st.error(f"**Unexpected error:** {exc}")
