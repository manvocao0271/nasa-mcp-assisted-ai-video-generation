"""Video Gen — calls Wan visual models on Qwen Cloud to generate scene clips (silent video only).

For each storyboard entry, submits an async job, polls until completion, and saves to output/clips/scene_N.mp4.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import httpx

OUTPUT_DIR = Path("output")
CLIPS_DIR = OUTPUT_DIR / "clips"

_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
_SUBMIT_URL = f"{_API_BASE}/services/aigc/video-generation/video-synthesis"
_TASK_URL = f"{_API_BASE}/tasks/{{task_id}}"

MODEL_T2V = "wan2.1-t2v-turbo"
MODEL_I2V = "wan2.7-i2v-2026-04-25"
MAX_POLL_SECONDS = 600
MAX_SCENES = 3


class VideoGen:
    """Wan visual-model client — one silent clip per storyboard entry."""

    MAX_DURATION = 10

    def __init__(self, qwen_api_key: str, poll_interval: float = 10.0) -> None:
        self.qwen_api_key = qwen_api_key
        self.poll_interval = poll_interval
        self._headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def run(
        self,
        storyboard: list[dict],
        on_clip_start: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Generate one clip per storyboard entry (up to MAX_SCENES)."""
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        entries = storyboard[:MAX_SCENES]
        total = len(entries)

        for i, entry in enumerate(entries):
            prompt = entry.get("prompt", "")
            if not prompt:
                continue

            if on_clip_start:
                on_clip_start(i + 1, total)

            ref_url = entry.get("ref_image_url", "")
            clip_path = self._unique_clip_path(entry["scene"])
            duration = min(entry.get("duration_seconds", 5), self.MAX_DURATION)
            task_id = self._submit_job(prompt, duration, ref_image_url=ref_url)
            video_url = self._poll_job(task_id)
            self._download_clip(video_url, clip_path)
            clips.append(clip_path)

        return clips

    def generate_one(self, entry: dict) -> Path:
        """Generate a single clip from one storyboard entry."""
        return self.run([entry])[0]

    @staticmethod
    def _unique_clip_path(scene: int) -> Path:
        candidate = CLIPS_DIR / f"scene_{scene}.mp4"
        counter = 1
        while candidate.exists():
            candidate = CLIPS_DIR / f"scene_{scene}_{counter}.mp4"
            counter += 1
        return candidate

    def _submit_job(self, prompt: str, duration_seconds: int, ref_image_url: str = "") -> str:
        use_ref = (
            ref_image_url
            and self._url_is_usable_image(ref_image_url)
            and self._ref_matches_prompt(ref_image_url, prompt)
        )
        # Append strict motion & lighting directives to avoid zooms and brightness shifts
        final_prompt = f"{prompt}\n\n{self._build_motion_directives(duration_seconds)}"

        if use_ref:
            model = MODEL_I2V
            inp: dict = {
                "prompt": final_prompt,
                "media": [{"type": "first_frame", "url": ref_image_url}],
            }
        else:
            model = MODEL_T2V
            inp = {"prompt": final_prompt}

        params = {
            "resolution": "720P",
            "prompt_extend": True,
        }

        if model == MODEL_I2V:
            params["duration"] = max(2, min(duration_seconds, 15))

        body = {
            "model": model,
            "input": inp,
            "parameters": params,
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(_SUBMIT_URL, json=body, headers=self._headers)
            if not resp.is_success:
                raise RuntimeError(
                    f"Video submit failed {resp.status_code}: {resp.text}"
                )
            data = resp.json()

        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in submit response: {data}")
        return task_id

    def _build_motion_directives(self, duration_seconds: int) -> str:
        """Return a short block of directives forcing orbit/rotation motion and constant lighting.

        These directives are purposely explicit to guide the video model:
        - Orbit/rotate the camera around the subject at a fixed distance (no zoom/scale change)
        - Keep the entire subject fully in frame for the whole clip
        - Use smooth, continuous rotation; avoid sudden jerks
        - Maintain constant exposure/brightness for all light sources; do NOT increase bloom,
          lens flare, or brightness of stars/galaxies during the clip
        - Keep styling photorealistic with minimal dynamic color grading
        """
        return (
            "VIDEO_DIRECTIVES: Camera behavior — orbit/rotate around the subject at a fixed distance; "
            "do NOT zoom in or out (no scale change). Keep the entire subject fully in frame for the entire clip. "
            "Motion: smooth, continuous rotation/orbit at constant speed; no sudden jerks or accelerations. "
            "Lighting: maintain constant exposure and brightness for all light sources; do NOT increase bloom, "
            "lens flares, or brightness of stars/galaxies during the duration. "
            "Style: photorealistic, minimal color grading. "
            f"Duration: {duration_seconds} seconds."
        )

    @staticmethod
    def _ref_matches_prompt(ref_url: str, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        url_lower = ref_url.lower()

        TOPIC_HINTS = {
            "mars": ["mars", "martian", "rover", "curiosity", "perseverance"],
            "earth": ["earth", "epic", "dscovr", "globe", "terra"],
            "moon": ["moon", "lunar", "apollo", "crater"],
            "asteroid": ["asteroid", "comet", "neo", "bennu"],
            "sun": ["sun", "solar", "corona", "flare"],
        }

        for topic, keywords in TOPIC_HINTS.items():
            prompt_is_about = any(k in prompt_lower for k in keywords)
            url_matches = any(k in url_lower for k in keywords)
            if prompt_is_about and not url_matches:
                return False

        return True

    @staticmethod
    def _url_is_usable_image(url: str) -> bool:
        _SUPPORTED = ("image/jpeg", "image/png", "image/gif", "image/webp")
        _WAN_BLOCKED_DOMAINS = ("images-assets.nasa.gov",)
        if any(blocked in url for blocked in _WAN_BLOCKED_DOMAINS):
            return False
        try:
            with httpx.Client(timeout=8) as client:
                r = client.head(url, follow_redirects=True)
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return r.status_code < 400 and ct in _SUPPORTED
        except Exception:
            return False

    def _poll_job(self, task_id: str) -> str:
        url = _TASK_URL.format(task_id=task_id)
        poll_headers = {"Authorization": f"Bearer {self.qwen_api_key}"}
        deadline = time.monotonic() + MAX_POLL_SECONDS

        while time.monotonic() < deadline:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=poll_headers)
                resp.raise_for_status()
                data = resp.json()

            output = data.get("output", {})
            status = output.get("task_status", "")

            if status == "SUCCEEDED":
                video_url = output.get("video_url")
                if not video_url:
                    raise RuntimeError(f"SUCCEEDED but no video_url in: {output}")
                return video_url
            elif status in ("FAILED", "CANCELED"):
                raise RuntimeError(f"Video generation {status}: {output}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Video generation timed out after {MAX_POLL_SECONDS}s (task={task_id})")

    def _download_clip(self, video_url: str, dest: Path) -> None:
        with httpx.stream("GET", video_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            dest.write_bytes(b"".join(r.iter_bytes()))
