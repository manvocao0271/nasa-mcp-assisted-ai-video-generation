"""Pale Blue Dot — AI Universe Video Generator

Streamlit chatbot UI. The user types a natural language astronomy request;
the pipeline streams status updates back to the browser as each agent runs,
then plays the finished video inline.

Launch:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import AuthenticationError, APIError

from agent.orchestrator import Orchestrator

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

# ── Sidebar — source panel ────────────────────────────────────────────────────

with st.sidebar:
    st.header("NASA Sources")
    sources_placeholder = st.empty()
    sources_placeholder.info("Sources will appear here after a run.")

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    "Get me any photo taken on Mars and generate a 10-second video of what it is like on Mars",
    "Make a short film about an asteroid passing close to Earth this week",
    "Show me the Astronomy Picture of the Day on July 20, 1969 and make a video about it",
    "Generate a video about an exoplanet similar to Earth",
]

if not QWEN_API_KEY:
    st.error("QWEN_API_KEY is not set. Add it to your .env file.")
    st.stop()

# Show retry toggle only when cached data exists
_cache_exists = (Path("output/assets.json").exists() or Path("output/script.json").exists())
resume_mode = st.sidebar.toggle(
    "♻️ Retry (use cached data)",
    value=False,
    disabled=not _cache_exists,
    help="Skip NASA fetch and Qwen script/storyboard calls — reuse the last run's cached outputs and go straight to video generation.",
)

user_input = st.chat_input("Ask anything about the universe…")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Pipeline status stream
        status_area = st.status("Running pipeline…", expanded=True)
        video_placeholder = st.empty()

        orchestrator = Orchestrator(
            qwen_api_key=QWEN_API_KEY,
            nasa_api_key=NASA_API_KEY,
        )

        final_video_path: Path | None = None
        manifest: dict = {}

        ICONS = {
            "data": "🛰️", "script": "✍️", "storyboard": "🎬",
            "video": "🎥", "edit": "✂️",
        }
        LABELS = {
            "data": "Fetching NASA data",
            "script": "Writing narration script",
            "storyboard": "Generating storyboard",
            "video": "Generating video clip (Wan 2.7)",
            "edit": "Assembling final film",
        }

        with status_area:
            step_placeholders: dict = {}
            try:
                for update in orchestrator.run(user_input, resume=resume_mode):
                    stage = update["stage"]
                    status = update["status"]
                    detail = update["detail"]
                    icon = ICONS.get(stage, "⚙️")
                    label = LABELS.get(stage, stage.title())

                    if stage == "done":
                        final_video_path = Path(detail) if detail else None
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

            except AuthenticationError:
                status_area.update(label="Authentication failed", state="error")
                st.error(
                    "**Invalid Qwen API key.** "
                    "Check that `QWEN_API_KEY` in your `.env` file contains a valid "
                    "[DashScope API key](https://dashscope.console.aliyun.com/apiKey) "
                    "(format: `sk-...`)."
                )
                st.stop()
            except Exception as exc:
                status_area.update(label="Pipeline error", state="error")
                st.error(f"**Pipeline error:** {exc}")
                st.stop()

        # Show finished video inline — assembled film or all accumulated clips
        clips_dir = Path("output/clips")
        # Sort by modification time so newest clips appear last
        scene_clips = (
            sorted(clips_dir.glob("scene_*.mp4"), key=lambda p: p.stat().st_mtime)
            if clips_dir.exists() else []
        )

        if final_video_path and final_video_path.exists():
            video_placeholder.video(str(final_video_path))
            response_text = f"Here's your film: **{manifest.get('title', 'Episode')}**"
        elif scene_clips:
            response_text = f"{len(scene_clips)} clip(s) in `output/clips/`:"
            with video_placeholder.container():
                for clip in scene_clips:
                    st.caption(f"Scene {clip.stem.split('_')[-1]}")
                    st.video(str(clip))
        else:
            response_text = (
                "Pipeline ran successfully. "
                "Script and storyboard written to `output/` — "
                "video clips will appear here once generated."
            )

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Update sidebar sources panel
        images = manifest.get("assets", {}).get("images", [])
        if images:
            with sources_placeholder.container():
                for img in images:
                    st.image(img["url"], caption=img.get("caption", ""), use_container_width=True)
                    st.caption(f"Source: {img.get('source', 'NASA')}")
        else:
            sources_placeholder.info("No images in this run.")
