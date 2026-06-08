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

        with status_area:
            for update in orchestrator.run(user_input):
                stage = update["stage"]
                detail = update["detail"]

                if stage == "done":
                    final_video_path = Path(detail)
                    manifest = update.get("manifest", {})
                    status_area.update(label="Pipeline complete!", state="complete")
                else:
                    icon = {
                        "data": "🛰️",
                        "script": "✍️",
                        "storyboard": "🎬",
                        "video": "🎥",
                        "edit": "✂️",
                    }.get(stage, "⚙️")
                    status_label = "done" if update["status"] == "done" else "running"
                    st.write(f"{icon} **{stage.title()}** — {detail}")

        # Show finished video inline
        if final_video_path and final_video_path.exists():
            video_placeholder.video(str(final_video_path))
            response_text = f"Here's your film: **{manifest.get('title', 'Episode')}**"
        else:
            response_text = "Pipeline finished. (Video generation not yet implemented.)"

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
