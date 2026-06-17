"""Storyboard Agent — one Wan visual prompt per script scene.

Each selected NASA image gets its own prompt and reference frame. No audio.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from agent.qwen_client import MODEL_VL_PLUS, QwenClient
from agent.video_gen import VideoGen

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """You write image-to-video prompts for Wan 2.7 (silent, visual only).
Describe what the camera sees — lighting, angle, subject, and motion. ≤80 tokens.
Do NOT include dialogue, narration, or on-screen text.
Return ONLY JSON: {"prompt": "..."}"""


class StoryboardAgent:
    """Generates one Wan prompt per scene, each locked to its NASA reference frame."""

    def __init__(self, qwen_api_key: str) -> None:
        self.qwen_api_key = qwen_api_key

    _SUPPORTED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    @classmethod
    def _url_is_usable_image(cls, url: str) -> bool:
        try:
            with httpx.Client(timeout=8) as client:
                r = client.head(url, follow_redirects=True)
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return r.status_code < 400 and ct in cls._SUPPORTED_IMAGE_TYPES
        except Exception:
            return False

    @classmethod
    def _fetch_image_as_data_uri(cls, url: str) -> str | None:
        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                r = client.get(url)
            if r.status_code >= 400:
                return None
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct not in cls._SUPPORTED_IMAGE_TYPES:
                return None
            b64 = base64.b64encode(r.content).decode()
            return f"data:{ct};base64,{b64}"
        except Exception:
            return None

    def run(self, script: dict, assets: dict, user_message: str, chat_description: str = "") -> list[dict]:
        client = QwenClient(self.qwen_api_key, model=MODEL_VL_PLUS)
        storyboard: list[dict] = []
        description_line = (
            f"Astronomer's description: {chat_description.strip()}\n"
            if chat_description.strip() else ""
        )

        for scene in script.get("scenes", []):
            scene_num = scene["scene"]
            ref_url = scene.get("ref_image_url", "")
            caption = scene.get("caption", "")
            mood = scene.get("mood", "cinematic")

            content: list = []
            if ref_url:
                data_uri = self._fetch_image_as_data_uri(ref_url)
                if data_uri:
                    content.append({"type": "image_url", "image_url": {"url": data_uri}})
                elif self._url_is_usable_image(ref_url):
                    content.append({"type": "image_url", "image_url": {"url": ref_url}})

            content.append({
                "type": "text",
                "text": (
                    f"User request: {user_message}\n"
                    f"{description_line}"
                    f"Scene {scene_num} mood: {mood}\n"
                    f"Scene caption: {caption}\n\n"
                    "Write the visual prompt JSON now."
                ),
            })

            response = client.chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ])
            raw = response.choices[0].message.content or ""
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())

            try:
                board = json.loads(raw)
                prompt = board.get("prompt", "")
            except json.JSONDecodeError:
                prompt = caption[:200] if caption else user_message[:120]

            storyboard.append({
                "scene": scene_num,
                "prompt": prompt,
                "ref_image_url": ref_url,
                "duration_seconds": VideoGen.MAX_DURATION,
            })

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "storyboard.json").write_text(json.dumps(storyboard, indent=2))
        return storyboard
