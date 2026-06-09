"""Script Agent — writes a 3-act narration script grounded in NASA assets.

Reads output/assets.json, calls Qwen to produce a structured screenplay,
and writes output/script.md.

Output schema (script dict):
    {
        "title": str,
        "scenes": [
            {
                "act": int,             # 1 | 2 | 3
                "narration": str,       # spoken narration (~100 words)
                "mood": str,            # cinematic mood descriptor
                "data_ref": str         # which assets.data key this scene draws from
            }
        ]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from agent.qwen_client import MODEL_PLUS, QwenClient

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """\
You are a science documentary scriptwriter. You will be shown real NASA satellite images \
alongside structured data about them. Write a 3-act short-film script (~300 words total) \
grounded strictly in what you see and the data provided. Be cinematic and factual.
Return ONLY valid JSON matching this schema exactly — no markdown fences, no extra text:
{
  "title": "<episode title>",
  "scenes": [
    {"act": 1, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"},
    {"act": 2, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"},
    {"act": 3, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"}
  ]
}
"""


class ScriptAgent:
    """Generates a 3-act narration script from NASA assets using Qwen (vision)."""

    def __init__(self, qwen_api_key: str) -> None:
        self.qwen_api_key = qwen_api_key

    _SUPPORTED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    @classmethod
    def _url_is_usable_image(cls, url: str) -> bool:
        """HEAD-check that the URL has Content-Length and a supported image Content-Type.

        Qwen's vision API rejects URLs without Content-Length and non-JPEG/PNG/GIF/WebP.
        """
        try:
            with httpx.Client(timeout=5) as client:
                r = client.head(url, follow_redirects=True)
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return "content-length" in r.headers and ct in cls._SUPPORTED_IMAGE_TYPES
        except Exception:
            return False

    def run(self, assets: dict, user_message: str) -> dict:
        """Generate script from assets. Returns script dict and writes output/script.md.

        Passes real NASA image URLs to Qwen as multimodal content so the model
        can visually ground the narration in what the images actually show.

        Args:
            assets: output from DataAgent.run()
            user_message: original user request for tone/context
        """
        client = QwenClient(self.qwen_api_key, model=MODEL_PLUS)

        # Build multimodal user message: images first, then structured data text
        content: list = []

        for img in assets.get("images", [])[:3]:  # cap at 3 images to stay within context
            url = img.get("url", "")
            if url and self._url_is_usable_image(url):
                content.append({"type": "image_url", "image_url": {"url": url}})

        # Summarise the structured data (strip heavy coordinate arrays to save tokens)
        data_summary = {}
        for key, val in assets.get("data", {}).items():
            if isinstance(val, dict):
                data_summary[key] = {
                    k: v for k, v in val.items()
                    if k not in ("dscovr_j2000_position", "lunar_j2000_position",
                                 "sun_j2000_position", "attitude_quaternions", "coords")
                }
            else:
                data_summary[key] = val

        content.append({
            "type": "text",
            "text": (
                f"User request: {user_message}\n\n"
                f"NASA data:\n{json.dumps(data_summary, indent=2)}\n\n"
                "Write the 3-act script JSON now."
            ),
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        response = client.chat(messages)
        raw = response.choices[0].message.content or ""

        # Strip markdown fences if model added them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            script: dict = json.loads(raw)
        except json.JSONDecodeError:            # Fallback: return a minimal valid structure so the pipeline continues
            script = {
                "title": user_message[:60],
                "scenes": [
                    {"act": i, "narration": "", "mood": "cinematic", "data_ref": ""}
                    for i in range(1, 4)
                ],
            }

        OUTPUT_DIR.mkdir(exist_ok=True)
        md_lines = [f"# {script.get('title', 'Untitled')}\n"]
        for scene in script.get("scenes", []):
            md_lines.append(f"## Act {scene['act']} — {scene['mood']}\n")
            md_lines.append(scene["narration"] + "\n")
        (OUTPUT_DIR / "script.md").write_text("\n".join(md_lines))
        (OUTPUT_DIR / "script.json").write_text(json.dumps(script, indent=2))

        return script
