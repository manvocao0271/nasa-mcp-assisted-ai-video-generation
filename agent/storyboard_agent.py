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
import re
from pathlib import Path

from agent.qwen_client import MODEL_PLUS, QwenClient

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """\
You are a cinematographer writing image-to-video prompts for Wan 2.7, a photorealistic \
video generation model. You will be shown real NASA satellite images and the narration \
script for each act. For each scene, write a single visual prompt of ≤80 tokens that \
describes exactly what the camera sees — lighting, camera angle, subject, and motion. \
Do NOT reproduce the narration; describe only the visuals.
Return ONLY a JSON array — no markdown fences, no extra text:
[
  {"scene": 1, "prompt": "...", "ref_image_url": "<pick the most relevant URL from images>"},
  {"scene": 2, "prompt": "...", "ref_image_url": "..."},
  {"scene": 3, "prompt": "...", "ref_image_url": "..."}
]
"""


class StoryboardAgent:
    """Generates Wan-compatible visual prompts with NASA reference frames."""

    def __init__(self, qwen_api_key: str) -> None:
        self.qwen_api_key = qwen_api_key

    def run(self, script: dict, assets: dict) -> list[dict]:
        """Generate one storyboard entry per scene.

        Calls Qwen with the script narrations + real NASA images so prompts are
        visually grounded. Returns storyboard list and writes output/storyboard.json.

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
            if url:
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
                + f"\n\nScript scenes:\n{scenes_text}\n\n"
                "Write the storyboard JSON array now."
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
            boards: list[dict] = json.loads(raw)
        except json.JSONDecodeError:
            boards = []

        # Merge with script metadata; fill gaps if model returned fewer entries
        scenes = script.get("scenes", [])
        storyboard: list[dict] = []
        for i, scene in enumerate(scenes):
            board = boards[i] if i < len(boards) else {}
            fallback_url = images[i]["url"] if i < len(images) else ""
            storyboard.append({
                "scene": i + 1,
                "act": scene["act"],
                "prompt": board.get("prompt", ""),
                "ref_image_url": board.get("ref_image_url", fallback_url),
                "duration_seconds": 5,
            })

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "storyboard.json").write_text(json.dumps(storyboard, indent=2))
        return storyboard
