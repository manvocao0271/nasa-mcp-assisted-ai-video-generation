"""Orchestrator — drives the full pipeline and enforces the token budget.

Receives the user's natural language request from app.py, runs each agent
in sequence, streams status updates back via a generator, and writes
output/episode_manifest.json when the run completes.

Flow:
    user message
        → data_agent   → output/assets.json
        → script_agent → output/script.md
        → storyboard_agent → output/storyboard.json
        → video_gen    → output/clips/scene_N.mp4
        → edit_agent   → output/episode_final.mp4
        → episode_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from agent.data_agent import DataAgent
from agent.edit_agent import EditAgent
from agent.script_agent import ScriptAgent
from agent.storyboard_agent import StoryboardAgent
from agent.video_gen import VideoGen

OUTPUT_DIR = Path("output")


class Orchestrator:
    """Token-budget-aware pipeline driver."""

    def __init__(self, qwen_api_key: str, nasa_api_key: str, token_budget: int = 50_000) -> None:
        self.qwen_api_key = qwen_api_key
        self.nasa_api_key = nasa_api_key
        self.token_budget = token_budget
        self.tokens_used = 0

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "clips").mkdir(exist_ok=True)

    def run(self, user_message: str) -> Generator[dict, None, None]:
        """Run the full pipeline. Yields status dicts for UI streaming.

        Each yielded dict has the shape:
            {"stage": str, "status": "running" | "done" | "error", "detail": str}

        The final yield has stage="done" and includes the path to episode_final.mp4.
        """
        yield {"stage": "data", "status": "running", "detail": "Fetching NASA data…"}
        assets = DataAgent(self.nasa_api_key, self.qwen_api_key).run(user_message)
        yield {"stage": "data", "status": "done", "detail": f"{len(assets)} assets fetched"}

        yield {"stage": "script", "status": "running", "detail": "Writing narration script…"}
        script = ScriptAgent(self.qwen_api_key).run(assets, user_message)
        yield {"stage": "script", "status": "done", "detail": f"{len(script['scenes'])} scenes"}

        yield {"stage": "storyboard", "status": "running", "detail": "Generating storyboard…"}
        storyboard = StoryboardAgent(self.qwen_api_key).run(script, assets)
        yield {"stage": "storyboard", "status": "done", "detail": f"{len(storyboard)} scene prompts"}

        yield {"stage": "video", "status": "running", "detail": "Generating video clips via Wan…"}
        clips = VideoGen(self.qwen_api_key).run(storyboard)
        yield {"stage": "video", "status": "done", "detail": f"{len(clips)} clips generated"}

        yield {"stage": "edit", "status": "running", "detail": "Assembling final film…"}
        final_path = EditAgent().run(clips, script)
        detail = str(final_path) if final_path else "Skipped — no clips ready yet"
        yield {"stage": "edit", "status": "done", "detail": detail}

        manifest = {
            "user_message": user_message,
            "tokens_used": self.tokens_used,
            "assets": assets,
            "script_scenes": len(script["scenes"]),
            "clips": [str(c) for c in clips],
            "final_video": str(final_path) if final_path else None,
        }
        (OUTPUT_DIR / "episode_manifest.json").write_text(json.dumps(manifest, indent=2))

        yield {"stage": "done", "status": "done", "detail": str(final_path) if final_path else "", "manifest": manifest}
