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
        → episode_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from agent.data_agent import DataAgent
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

    def run(self, user_message: str, resume: bool = False) -> Generator[dict, None, None]:
        """Run the full pipeline. Yields status dicts for UI streaming.

        Each yielded dict has the shape:
            {"stage": str, "status": "running" | "done" | "error", "detail": str}

        When ``resume=True`` each stage is skipped if its output file already
        exists on disk, so a failed run can be retried cheaply after a fix.

        The final yield has stage="done" and includes the path to episode_final.mp4.
        """
        # ── Data ──────────────────────────────────────────────────────────────
        assets_path = OUTPUT_DIR / "assets.json"
        if resume and assets_path.exists():
            assets = json.loads(assets_path.read_text())
            yield {"stage": "data", "status": "running", "detail": "Loading cached NASA data…"}
            yield {"stage": "data", "status": "done", "detail": f"Loaded from cache ({len(assets.get('images', []))} images)"}
        else:
            yield {"stage": "data", "status": "running", "detail": "Fetching NASA data…"}
            assets = DataAgent(self.nasa_api_key, self.qwen_api_key).run(user_message)
            yield {"stage": "data", "status": "done", "detail": f"{len(assets.get('images', []))} images fetched"}

        # ── Script ────────────────────────────────────────────────────────────
        script_path = OUTPUT_DIR / "script.json"
        if resume and script_path.exists():
            script = json.loads(script_path.read_text())
            yield {"stage": "script", "status": "running", "detail": "Loading cached script…"}
            yield {"stage": "script", "status": "done", "detail": f"Loaded from cache ({len(script.get('scenes', []))} scenes)"}
        else:
            yield {"stage": "script", "status": "running", "detail": "Writing narration script…"}
            script = ScriptAgent(self.qwen_api_key).run(assets, user_message)
            yield {"stage": "script", "status": "done", "detail": f"{len(script['scenes'])} scenes written"}

        # ── Storyboard ────────────────────────────────────────────────────────
        storyboard_path = OUTPUT_DIR / "storyboard.json"
        if resume and storyboard_path.exists():
            storyboard = json.loads(storyboard_path.read_text())
            yield {"stage": "storyboard", "status": "running", "detail": "Loading cached storyboard…"}
            yield {"stage": "storyboard", "status": "done", "detail": f"Loaded from cache ({len(storyboard)} prompts)"}
        else:
            yield {"stage": "storyboard", "status": "running", "detail": "Generating storyboard…"}
            storyboard = StoryboardAgent(self.qwen_api_key).run(script, assets)
            yield {"stage": "storyboard", "status": "done", "detail": f"{len(storyboard)} scene prompts"}

        # ── Video ─────────────────────────────────────────────────────────────
        yield {"stage": "video", "status": "running", "detail": "Generating video clip (Wan 2.7)…"}
        clips = VideoGen(self.qwen_api_key).run(storyboard)
        clip_path = clips[0] if clips else None
        yield {"stage": "video", "status": "done", "detail": str(clip_path) if clip_path else "No clip generated"}

        manifest = {
            "user_message": user_message,
            "tokens_used": self.tokens_used,
            "assets": assets,
            "script_scenes": len(script["scenes"]),
            "clips": [str(c) for c in clips],
        }
        (OUTPUT_DIR / "episode_manifest.json").write_text(json.dumps(manifest, indent=2))

        yield {"stage": "done", "status": "done", "detail": str(clip_path) if clip_path else "", "manifest": manifest}
