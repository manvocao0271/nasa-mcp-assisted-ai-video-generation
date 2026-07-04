"""Script Agent — writes one scene per selected NASA reference image.

Reads output/assets.json, calls Qwen to produce short visual captions grounded in each frame, and writes output/script.json. No audio or narration — captions guide the storyboard only.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from agent.qwen_client import MODEL_VL_PLUS, QwenClient

_DEFAULT_OUTPUT_DIR = Path("output")
MAX_SCENES = 3


def scene_count(assets: dict) -> int:
    """One scene per selected image; one text-only scene when no images."""
    n = len(assets.get("images", []))
    return max(1, min(n, MAX_SCENES))


def _build_system_prompt(n: int) -> str:
    return f"""You are a science documentary writer. You will see real NASA images and data.
Write exactly {n} scene(s) for a silent short video — no voiceover, no dialogue.
Each scene needs a short visual caption (~2 sentences) describing what the viewer sees, tied to the user's request.
If an astronomer's description is provided, use its specific visual details (colours, textures, scale) to enrich the captions.
Return ONLY valid JSON — no markdown fences:
{{
  "title": "<short title>",
  "scenes": [
    {{"scene": 1, "caption": "...", "mood": "...", "data_ref": "<key from assets.data>"}}
  ]
}}
"""


class ScriptAgent:
    """Generates one caption scene per selected NASA image using Qwen (vision)."""

    def __init__(self, qwen_api_key: str, output_dir: Path | None = None) -> None:
        self.qwen_api_key = qwen_api_key
        self.output_dir = output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR

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

    def run(self, assets: dict, user_message: str, chat_description: str = "") -> dict:
        images = assets.get("images", [])[:MAX_SCENES]
        n = scene_count(assets)
        selected_urls = [img["url"] for img in images if img.get("url")]

        client = QwenClient(self.qwen_api_key, model=MODEL_VL_PLUS)
        content: list = []

        for i, img in enumerate(images, start=1):
            url = img.get("thumb_url") or img.get("url", "")
            if not url:
                continue
            data_uri = self._fetch_image_as_data_uri(url)
            if data_uri:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif self._url_is_usable_image(url):
                content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({
                "type": "text",
                "text": f"Image {i} caption hint: {img.get('caption', '')[:120]}",
            })

        data_summary = {}
        for key, val in assets.get("data", {}).items():
            if isinstance(val, dict):
                data_summary[key] = {
                    k: v for k, v in val.items()
                    if k not in (
                        "dscovr_j2000_position", "lunar_j2000_position",
                        "sun_j2000_position", "attitude_quaternions", "coords",
                    )
                }
            else:
                data_summary[key] = val

        description_block = (
            f"\nAstronomer's description of the subject:\n{chat_description.strip()}\n"
            if chat_description.strip() else ""
        )
        image_note = (
            f"There are {n} image(s). Write scene i about Image i."
            if selected_urls
            else "No images — write one scene from the NASA data alone."
        )
        content.append({
            "type": "text",
            "text": (
                f"User request: {user_message}\n"
                f"{description_block}\n"
                f"NASA data:\n{json.dumps(data_summary, indent=2)}\n\n"
                f"{image_note}\n"
                f"Write exactly {n} scene(s) as JSON now."
            ),
        })

        messages = [
            {"role": "system", "content": _build_system_prompt(n)},
            {"role": "user", "content": content},
        ]

        response = client.chat(messages)
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            script: dict = json.loads(raw)
        except json.JSONDecodeError:
            script = {"title": user_message[:60], "scenes": []}

        script = self._normalize(script, user_message, n, images, selected_urls)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        md_lines = [f"# {script.get('title', 'Untitled')}\n"]
        for scene in script["scenes"]:
            md_lines.append(f"## Scene {scene['scene']} \u2014 {scene['mood']}\n")
            md_lines.append(scene["caption"] + "\n")
        (self.output_dir / "script.md").write_text("\n".join(md_lines))
        (self.output_dir / "script.json").write_text(json.dumps(script, indent=2))

        return script

    @staticmethod
    def _normalize(
        script: dict,
        user_message: str,
        n: int,
        images: list[dict],
        selected_urls: list[str],
    ) -> dict:
        scenes = script.get("scenes", [])
        normalized: list[dict] = []

        for i in range(n):
            src = scenes[i] if i < len(scenes) else {}
            ref_url = (images[i].get("thumb_url") or selected_urls[i]) if i < len(selected_urls) else ""
            normalized.append({
                "scene": i + 1,
                "caption": src.get("caption", ""),
                "mood": src.get("mood", "cinematic"),
                "ref_image_url": ref_url,
                "data_ref": src.get("data_ref", images[i].get("source", "") if i < len(images) else ""),
            })

        return {
            "title": script.get("title", user_message[:60]),
            "user_message": user_message,
            "scene_count": n,
            "selected_image_urls": selected_urls,
            "scenes": normalized,
        }
