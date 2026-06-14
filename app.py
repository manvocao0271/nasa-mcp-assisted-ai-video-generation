"""Pale Blue Dot — AI Universe Video Generator

Streamlit chatbot UI. The user types a natural language astronomy request; the pipeline streams status updates back to the browser as each agent runs, then plays the finished video inline.

Launch:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import AuthenticationError, APIError

from agent.orchestrator import Orchestrator
from agent.chat_agent import ChatAgent
from agent.qwen_client import QwenClient
from agent.run_db import RunDB, Message

load_dotenv()

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pale Blue Dot",
    page_icon="🌍",
    layout="centered",
)

st.title("🌌 Pale Blue Dot")
st.caption("Type an astronomy request and watch a short film get built in real time.")

# ── Sidebar — conversation and source panel ──────────────────────────────────

with st.sidebar:
    tab_chat, tab_history = st.tabs(["Chat", "History"])

    with tab_chat:
        st.header("NASA Sources")
        sources_placeholder = st.empty()
        sources_placeholder.info("Sources will appear here after a run.")

    with tab_history:
        st.header("Past Conversations")
        conversations = st.session_state.run_db.list_conversations()
        if conversations:
            for conv in conversations:
                conv_title = conv.get("title") or "Untitled"
                conv_date = conv["created_at"][:10] if conv["created_at"] else ""
                if st.button(f"📍 {conv_title}\n_{conv_date}_", use_container_width=True):
                    st.session_state.conversation_id = conv["conversation_id"]
                    # Load messages from conversation history
                    history = st.session_state.run_db.get_conversation_history(conv["conversation_id"])
                    st.session_state.messages = []
                    for run in history:
                        st.session_state.messages.append({"role": "user", "content": run["user_message"]})
                        st.session_state.messages.append({"role": "assistant", "content": run["assistant_response"]})
                    st.session_state.phase = "idle"
                    st.rerun()
        else:
            st.info("No past conversations yet.")

# ── Session state ─────────────────────────────────────────────────────────────
# phase:
#   "idle"              – waiting for user input, showing chat
#   "video_request"     – user wants to generate video, fetching NASA data
#   "image_selection"   – NASA data fetched, waiting for user to pick images
#   "pipeline"          – user confirmed selection, pipeline running

if "run_db" not in st.session_state:
    st.session_state.run_db = RunDB()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

for key, default in [
    ("messages", []),
    ("phase", "idle"),
    ("pending_message", ""),
    ("pending_assets", {}),
    ("resume_for_pipeline", False),
    ("video_topic", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Chat history replay ───────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Helpers ───────────────────────────────────────────────────────────────────

ICONS = {
    "data": "🛰️", "script": "✍️", "storyboard": "🎬", "video": "🎥",
}
LABELS = {
    "data":       "Fetching NASA data",
    "script":     "Writing scene captions",
    "storyboard": "Generating storyboard",
    "video":      "Generating video clips (Wan 2.7)",
}


def _fetch_thumb(url: str) -> bytes | str:
    """Return image bytes for domains that block browser access, else the original URL.

    NASA Image Library assets (images-assets.nasa.gov) return 403/timeout when loaded directly by the browser. We fetch them server-side and hand Streamlit raw bytes instead, which always works.
    """

    _PROXY_DOMAINS = ("images-assets.nasa.gov",)
    if not any(d in url for d in _PROXY_DOMAINS):
        return url  # fast path — most URLs are fine
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            r = client.get(url)
        if r.status_code < 400:
            return r.content
    except Exception:
        pass
    return url  # fall back to URL on failure; Streamlit will show broken-image icon


def _render_pipeline_stages(generator, status_area) -> dict:
    """Consume a pipeline generator, updating the status area in real time.

    Returns the manifest dict from the "done" event (or {} if not reached).
    """
    
    step_placeholders: dict = {}
    manifest: dict = {}
    with status_area:
        for update in generator:
            stage  = update["stage"]
            status = update["status"]
            detail = update["detail"]
            icon   = ICONS.get(stage, "⚙️")
            label  = LABELS.get(stage, stage.title())

            if stage == "done":
                manifest = update.get("manifest", {})
                status_area.update(label="Pipeline complete!", state="complete")
            elif status == "running":
                ph = st.empty()
                step_placeholders[stage] = ph
                ph.markdown(f"{icon} **{label}** — ⏳ *running…*")
                status_area.update(label=f"{icon} {label}…")
            elif status == "done":
                ph = step_placeholders.get(stage, st.empty())
                ph.markdown(f"{icon} **{label}** — ✅ {detail}")
    return manifest


def _show_clips() -> list[Path]:
    clips_dir = Path("output/clips")
    return (
        sorted(clips_dir.glob("scene_*.mp4"), key=lambda p: p.stat().st_mtime)
        if clips_dir.exists() else []
    )


def _update_sidebar(images: list[dict]) -> None:
    if images:
        with sources_placeholder.container():
            for img in images:
                st.image(_fetch_thumb(img["url"]), caption=img.get("caption", ""), use_container_width=True)
                st.caption(f"Source: {img.get('source', 'NASA')}")
    else:
        sources_placeholder.info("No images in this run.")


# ── Guards ────────────────────────────────────────────────────────────────────

if not QWEN_API_KEY:
    st.error("QWEN_API_KEY is not set. Add it to your .env file.")
    st.stop()

# Show retry toggle only when cached data exists
_cache_exists = (Path("output/assets.json").exists() or Path("output/script.json").exists())
resume_mode = st.sidebar.toggle(
    "♻️ Retry (use cached data)",
    value=False,
    disabled=not _cache_exists,
    help="Skip NASA fetch and Qwen script/storyboard calls — reuse the last run's cached outputs.",
)

# ── PHASE: idle — chat with context ─────────────────────────────────────────

user_input = st.chat_input(
    "Ask anything about the universe…",
    disabled=(st.session_state.phase != "idle"),
)

if user_input and st.session_state.phase == "idle":
    st.session_state.pending_message = user_input

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Chat with context from conversation history
    try:
        qwen_client = QwenClient(api_key=QWEN_API_KEY)
        chat_agent = ChatAgent(qwen_client)

        # Load conversation history for context
        history = st.session_state.run_db.get_conversation_history(st.session_state.conversation_id)

        with st.chat_message("assistant"):
            # Get response from ChatAgent
            result = chat_agent.answer(user_input, history)
            answer = result["answer"]
            should_generate = result["should_generate_video"]
            video_topic = result.get("video_topic", user_input)

            # Display answer
            st.markdown(answer)
            # Display retrieved passages (if any) produced by the retriever
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
                # Add retrieval info to conversation history so it persists
                passages_text = "\n".join([
                    f"{p.get('source')}: {p.get('snippet','').strip()} ({p.get('doc_id','')})"
                    for p in retrieved
                ])
                st.session_state.messages.append({"role": "assistant", "content": "Retrieved sources:\n" + passages_text})
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show video generation button if suggested
            if should_generate:
                st.session_state.video_topic = video_topic
                if st.button("🎬 Generate Video", type="primary"):
                    st.session_state.phase = "video_request"
                    st.rerun()

    except (AuthenticationError, APIError) as exc:
        st.error(f"**Chat error:** {exc}")
        st.session_state.phase = "idle"
        st.stop()
    except Exception as exc:
        st.error(f"**Unexpected error:** {exc}")
        st.session_state.phase = "idle"
        st.stop()

# ── PHASE: video_request — fetch data for video generation ──────────────────

if st.session_state.phase == "video_request":
    user_message = st.session_state.pending_message
    video_topic = st.session_state.video_topic

    with st.chat_message("assistant"):
        status_area = st.status("Fetching NASA data for video…", expanded=True)
        assets: dict = {}
        try:
            orchestrator = Orchestrator(qwen_api_key=QWEN_API_KEY, nasa_api_key=NASA_API_KEY)
            with status_area:
                for update in orchestrator.fetch_data(video_topic, resume=False):
                    assets = update.get("assets", assets)
                    stage  = update["stage"]
                    status = update["status"]
                    detail = update["detail"]
                    icon   = ICONS.get(stage, "⚙️")
                    label  = LABELS.get(stage, stage.title())
                    if status == "running":
                        ph = st.empty()
                        ph.markdown(f"{icon} **{label}** — ⏳ *running…*")
                        status_area.update(label=f"{icon} {label}…")
                    elif status == "done":
                        ph.markdown(f"{icon} **{label}** — ✅ {detail}")
                status_area.update(
                    label=f"✅ {len(assets.get('images', []))} NASA images fetched — pick below",
                    state="complete",
                )
        except (AuthenticationError, APIError, Exception) as exc:
            status_area.update(label="Data fetch failed", state="error")
            st.error(f"**Error fetching NASA data:** {exc}")
            st.session_state.phase = "idle"
            st.stop()

        st.session_state.pending_assets = assets
        st.session_state.phase = "image_selection"
        st.rerun()

# ── PHASE: image_selection — image picker ───────────────────────────────────

if st.session_state.phase == "image_selection":
    images = st.session_state.pending_assets.get("images", [])

    with st.chat_message("assistant"):
        if not images:
            st.warning("No images were returned by NASA tools. Generating video from text only.")
            if st.button("🎬 Generate Video (text only)"):
                st.session_state.phase = "pipeline"
                st.rerun()
        else:
            st.markdown(
                f"**Found {len(images)} NASA image{'s' if len(images) != 1 else ''}.**  "
                "Select the ones you want to use as visual reference for the video, then click **Generate**."
            )

            # Image selection grid — up to 3 columns
            n_cols = min(len(images), 3)
            cols = st.columns(n_cols)
            for i, img in enumerate(images):
                with cols[i % n_cols]:
                    thumb = _fetch_thumb(img.get("thumb_url") or img["url"])
                    st.image(thumb, use_container_width=True)
                    caption = img.get("caption") or img.get("title") or ""
                    if caption:
                        st.caption(caption[:80])
                    st.checkbox(
                        "Use this image",
                        value=True,
                        key=f"img_check_{i}",
                        label_visibility="visible",
                    )

            col_btn, col_all, col_none = st.columns([2, 1, 1])
            generate_clicked = col_btn.button("🎬 Generate Video", type="primary", use_container_width=True)
            if col_all.button("Select all", use_container_width=True):
                for i in range(len(images)):
                    st.session_state[f"img_check_{i}"] = True
                st.rerun()
            if col_none.button("Select none", use_container_width=True):
                for i in range(len(images)):
                    st.session_state[f"img_check_{i}"] = False
                st.rerun()

            if generate_clicked:
                selected = [
                    img for i, img in enumerate(images)
                    if st.session_state.get(f"img_check_{i}", True)
                ]
                # Fallback: if user deselected everything, use all images
                if not selected:
                    selected = images
                # Overwrite images list with the user's selection
                filtered_assets = dict(st.session_state.pending_assets)
                filtered_assets["images"] = selected
                st.session_state.pending_assets = filtered_assets
                st.session_state.phase = "pipeline"
                st.rerun()

# ── PHASE: pipeline — run the rest of the pipeline ──────────────────────────

if st.session_state.phase == "pipeline":
    user_message  = st.session_state.pending_message
    assets        = st.session_state.pending_assets
    resume        = st.session_state.resume_for_pipeline
    orchestrator  = Orchestrator(qwen_api_key=QWEN_API_KEY, nasa_api_key=NASA_API_KEY)

    with st.chat_message("assistant"):
        status_area = st.status("Running pipeline…", expanded=True)
        video_placeholder = st.empty()
        manifest: dict = {}

        try:
            manifest = _render_pipeline_stages(
                orchestrator.run_pipeline(user_message, assets, resume=resume),
                status_area,
            )
        except (AuthenticationError, APIError) as exc:
            status_area.update(label="Authentication failed", state="error")
            st.error(f"**Invalid Qwen API key:** {exc}")
            st.session_state.phase = "idle"
            st.stop()
        except Exception as exc:
            status_area.update(label="Pipeline error", state="error")
            exc_str = str(exc)
            if "AllocationQuota.FreeTierOnly" in exc_str:
                st.error(
                    "**Video generation quota exhausted.**  "
                    "The free tier for this model has been used up.  \n"
                    "To continue: open the [DashScope console](https://dashscope.console.aliyun.com/) "
                    "→ **Model Service** → disable **\"Use free tier only\"** mode to enable paid access."
                )
            else:
                st.error(f"**Pipeline error:** {exc}")
            st.session_state.phase = "idle"
            st.stop()

        # Show clips from this run (manifest), not older files in output/clips/
        manifest_clips = [Path(p) for p in manifest.get("clips", []) if Path(p).exists()]
        scene_clips = manifest_clips or _show_clips()
        if scene_clips:
            response_text = f"{len(scene_clips)} clip(s) generated:"
            with video_placeholder.container():
                for clip in scene_clips:
                    st.caption(clip.name)
                    st.video(str(clip))
        else:
            response_text = (
                "Script and storyboard written to `output/` — "
                "video clips will appear here once generated."
            )

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Save run to database
        run_id = str(uuid.uuid4())
        st.session_state.run_db.save_run(
            run_id=run_id,
            conversation_id=st.session_state.conversation_id,
            user_message=user_message,
            assistant_response=response_text,
            assets=assets,
            manifest=manifest,
            messages=[
                Message(role=m["role"], content=m["content"], timestamp=datetime.now().isoformat())
                for m in st.session_state.messages
            ],
        )

        # Update sidebar
        _update_sidebar(assets.get("images", []))

    st.session_state.phase = "idle"

