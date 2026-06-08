"""Storyboard Agent — converts each script scene into a video generation prompt.

Reads the script dict and assets, calls Qwen to produce a compact visual
prompt (≤80 tokens) per scene, and pairs each prompt with a NASA reference
image URL to use as the Wan image2video style anchor.

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

import json
from pathlib import Path

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """\
You are a cinematographer writing image-to-video prompts for Wan (a video generation model).
For each scene, write a single visual prompt of ≤80 tokens that describes what the camera sees.
Be specific about lighting, camera angle, subject, and motion. Do not include narration text.
Return a JSON array with one object per scene:
[{"scene": 1, "prompt": "...", "ref_image_url": "<url from assets.images>"}, ...]
"""


class StoryboardAgent:
    """Generates Wan-compatible visual prompts with NASA reference frames."""

    def __init__(self, qwen_api_key: str) -> None:
        self.qwen_api_key = qwen_api_key

    def run(self, script: dict, assets: dict) -> list[dict]:
        """Generate one storyboard entry per scene.

        Args:
            script: output from ScriptAgent.run()
            assets: output from DataAgent.run()

        Returns list of storyboard dicts and writes output/storyboard.json.
        """
        # TODO: call Qwen with SYSTEM_PROMPT + script scenes + assets.images,
        # parse JSON array response, fill in duration_seconds (default 10).
        storyboard = [
            {
                "scene": i + 1,
                "act": scene["act"],
                "prompt": "",
                "ref_image_url": assets["images"][i]["url"] if i < len(assets["images"]) else "",
                "duration_seconds": 10,
            }
            for i, scene in enumerate(script["scenes"])
        ]

        (OUTPUT_DIR / "storyboard.json").write_text(json.dumps(storyboard, indent=2))
        return storyboard
