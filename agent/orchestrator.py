"""Orchestrator — drives the full pipeline and enforces the token budget.

Flow:
    user message → data_agent → assets.json
    user picks images → script_agent → script.json
    → storyboard_agent → storyboard.json
    → video_gen → clips/scene_N.mp4 (one per selected image)
    → episode_manifest.json
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Generator

from agent.data_agent import DataAgent
from agent.script_agent import ScriptAgent, scene_count
from agent.storyboard_agent import StoryboardAgent
from agent.video_gen import VideoGen

OUTPUT_DIR = Path("output")


def _selected_urls(assets: dict) -> list[str]:
    return [img["url"] for img in assets.get("images", []) if img.get("url")]


def _script_cache_valid(script: dict, assets: dict) -> bool:
    urls = _selected_urls(assets)
    return (
        script.get("selected_image_urls") == urls
        and len(script.get("scenes", [])) == scene_count(assets)
    )


class Orchestrator:
    """Token-budget-aware pipeline driver."""

    def __init__(
        self,
        qwen_api_key: str,
        nasa_api_key: str,
        token_budget: int = 50_000,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.qwen_api_key = qwen_api_key
        self.nasa_api_key = nasa_api_key
        self.token_budget = token_budget
        self.tokens_used = 0
        self.cancel_event = cancel_event

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "clips").mkdir(exist_ok=True)

    @staticmethod
    def _cache_matches_topic(assets: dict, user_message: str) -> bool:
        """Return True if the cached assets are relevant to the requested topic.

        Uses simple keyword overlap: at least one significant word from the
        user message must appear in the cached query string (case-insensitive).
        Single-word queries must match exactly.
        """
        cached_query = (assets.get("query") or "").lower()
        if not cached_query:
            return False
        # Extract words longer than 3 chars to skip stop words
        topic_words = {w for w in user_message.lower().split() if len(w) > 3}
        if not topic_words:
            return cached_query == user_message.lower()
        return any(w in cached_query for w in topic_words)

    def fetch_data(self, user_message: str, resume: bool = False, context: str = "") -> Generator[dict, None, None]:
        """Fetch NASA assets for *user_message*.

        Args:
            user_message: The primary search topic (ideally already distilled to
                the specific subject, e.g. \"WASP-76b ultra-hot Jupiter\").
            resume: If True, reuse a cached assets.json when the topic matches.
            context: Optional extra text (e.g. the assistant's prior answer) that
                helps the DataAgent pick the right NASA tools and search terms.
        """
        assets_path = OUTPUT_DIR / "assets.json"
        if resume and assets_path.exists():
            assets = json.loads(assets_path.read_text())
            if self._cache_matches_topic(assets, user_message):
                yield {"stage": "data", "status": "running", "detail": "Loading cached NASA data…"}
                yield {"stage": "data", "status": "done",
                       "detail": f"Loaded from cache ({len(assets.get('images', []))} images)",
                       "assets": assets}
                return
            # Cache is stale (different topic) — fall through to a fresh fetch
            fetch_detail = "Cached data is for a different topic — fetching fresh NASA data…"
        else:
            fetch_detail = "Fetching NASA data…"

        yield {"stage": "data", "status": "running", "detail": fetch_detail}
        assets = DataAgent(self.nasa_api_key, self.qwen_api_key).run(user_message, context=context)
        yield {"stage": "data", "status": "done",
               "detail": f"{len(assets.get('images', []))} images fetched",
               "assets": assets}

    def run_pipeline(self, user_message: str, assets: dict, resume: bool = False, chat_description: str = "") -> Generator[dict, None, None]:
        n = scene_count(assets)
        urls = _selected_urls(assets)

        # ── Script ────────────────────────────────────────────────────────────
        script_path = OUTPUT_DIR / "script.json"
        use_script_cache = resume and script_path.exists()
        if use_script_cache:
            script = json.loads(script_path.read_text())
            use_script_cache = _script_cache_valid(script, assets)

        if use_script_cache:
            yield {"stage": "script", "status": "running", "detail": "Loading cached script…"}
            yield {"stage": "script", "status": "done",
                   "detail": f"Loaded from cache ({len(script.get('scenes', []))} scenes)"}
        else:
            yield {"stage": "script", "status": "running", "detail": f"Writing {n} scene caption(s)…"}
            script = ScriptAgent(self.qwen_api_key).run(assets, user_message, chat_description=chat_description)
            yield {"stage": "script", "status": "done",
                   "detail": f"{len(script['scenes'])} scene(s) written"}

        # ── Storyboard ────────────────────────────────────────────────────────
        storyboard_path = OUTPUT_DIR / "storyboard.json"
        use_board_cache = resume and storyboard_path.exists()
        if use_board_cache:
            storyboard = json.loads(storyboard_path.read_text())
            use_board_cache = len(storyboard) == len(script.get("scenes", []))

        if use_board_cache:
            yield {"stage": "storyboard", "status": "running", "detail": "Loading cached storyboard…"}
            yield {"stage": "storyboard", "status": "done",
                   "detail": f"Loaded from cache ({len(storyboard)} prompts)"}
        else:
            yield {"stage": "storyboard", "status": "running", "detail": "Generating storyboard…"}
            storyboard = StoryboardAgent(self.qwen_api_key).run(script, assets, user_message, chat_description=chat_description)
            yield {"stage": "storyboard", "status": "done",
                   "detail": f"{len(storyboard)} scene prompt(s)"}

        # ── Video ─────────────────────────────────────────────────────────────
        total = len(storyboard)

        yield {"stage": "video", "status": "running",
               "detail": f"Generating clip 1/{total}…" if total else "No scenes to render"}

        video_gen = VideoGen(self.qwen_api_key, cancel_event=self.cancel_event)
        clips: list[Path] = []

        for i, entry in enumerate(storyboard):
            if i > 0:
                yield {"stage": "video", "status": "running",
                       "detail": f"Generating clip {i + 1}/{total}…"}
            clip = video_gen.generate_one(entry)
            clips.append(clip)
            for w in video_gen.warnings:
                yield {"stage": "warning", "status": "warning", "detail": w}
            video_gen.warnings.clear()

        detail = f"{len(clips)} clip(s) generated" if clips else "No clips generated"
        yield {"stage": "video", "status": "done", "detail": detail}

        manifest = {
            "user_message": user_message,
            "scene_count": n,
            "selected_image_urls": urls,
            "tokens_used": self.tokens_used,
            "assets": assets,
            "clips": [str(c) for c in clips],
        }
        (OUTPUT_DIR / "episode_manifest.json").write_text(json.dumps(manifest, indent=2))

        yield {"stage": "done", "status": "done", "detail": detail, "manifest": manifest}

    def run(self, user_message: str, resume: bool = False) -> Generator[dict, None, None]:
        assets: dict = {}
        for update in self.fetch_data(user_message, resume=resume):
            assets = update.get("assets", assets)
            yield update
        yield from self.run_pipeline(user_message, assets, resume=resume)
