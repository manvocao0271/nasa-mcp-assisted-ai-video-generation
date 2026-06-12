"""Storyboard Agent — converts each script scene into a video generation prompt.

Reads the script dict and assets, calls Qwen to produce a compact visual prompt (≤80 tokens) per scene, and pairs each prompt with a NASA reference image URL to use as the Wan image2video style anchor.

Output schema (output/storyboard.json):
    [
        {
            "scene": int,               # 1-indexed
            "act": int,
            "prompt": str,              # ≤80 tokens, Wan/HappyHorse compatible
            "ref_image_url": str,       # NASA image URL for style anchoring
            "duration_seconds": int     # target clip length (default 10)
        }
    ]
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from agent.qwen_client import MODEL_PLUS, QwenClient
from agent.video_gen import VideoGen

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """
You are a cinematographer writing an image-to-video prompt for Wan 2.7, a photorealistic video generation model. You will be shown real NASA images and a narration script. Write a single visual prompt of ≤80 tokens describing what the camera sees — lighting, camera angle, subject, and motion. Do NOT reproduce narration text; describe only the visuals. Pick the most relevant NASA image URL as the reference frame. Return ONLY a JSON object — no markdown fences, no extra text:
{"prompt": "...", "ref_image_url": "<most relevant URL from the images provided>"}
"""


class StoryboardAgent:
    """Generates Wan-compatible visual prompts with NASA reference frames."""

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
        """Download *url* and return a base64 data URI Qwen can always ingest."""
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

    def run(self, script: dict, assets: dict) -> list[dict]:
        """Generate one storyboard entry per scene.

        Calls Qwen with the script narrations + real NASA images so prompts are visually grounded. Returns storyboard list and writes output/storyboard.json.

        Args:
            script: output from ScriptAgent.run()
            assets: output from DataAgent.run()
        """
        client = QwenClient(self.qwen_api_key, model=MODEL_PLUS)
        images = assets.get("images", [])

        # Build multimodal content: images first, then scene descriptions
        content: list = []
        for img in images[:3]:
            url = img.get("url", "")
            if not url:
                continue
            data_uri = self._fetch_image_as_data_uri(url)
            if data_uri:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif self._url_is_usable_image(url):
                content.append({"type": "image_url", "image_url": {"url": url}})

        scenes_text = "\n".join(
            f"Scene {s['act']}: {s['narration'][:200]}"
            for s in script.get("scenes", [])
        )
        image_urls = [img["url"] for img in images if img.get("url")]

        content.append({
            "type": "text",
            "text": (
                f"Available NASA image URLs:\n"
                + "\n".join(f"  - {u}" for u in image_urls)
                + f"\n\nScript (all acts combined):\n{scenes_text}\n\n"
                "Write the storyboard JSON object now."
            ),
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        response = client.chat(messages)
        raw = response.choices[0].message.content or ""

        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            board: dict = json.loads(raw)
        except json.JSONDecodeError:
            board = {}

        fallback_url = images[0]["url"] if images else ""
        storyboard = [{
            "scene": 1,
            "act": 1,
            "prompt": board.get("prompt", ""),
            "ref_image_url": board.get("ref_image_url", fallback_url),
            "duration_seconds": VideoGen.MAX_DURATION,
        }]

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "storyboard.json").write_text(json.dumps(storyboard, indent=2))
        return storyboard
