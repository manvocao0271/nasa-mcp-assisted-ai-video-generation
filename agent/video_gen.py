"""Video Gen — calls Wan 2.7 on Qwen Cloud to generate scene clips.

For each storyboard entry, submits an async text-to-video job to the
DashScope international API, polls until completion, downloads the MP4,
and saves it to output/clips/scene_N.mp4.

API reference: https://docs.qwencloud.com/developer-guides/video-generation/text-to-video
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

OUTPUT_DIR = Path("output")
CLIPS_DIR = OUTPUT_DIR / "clips"

_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
_SUBMIT_URL = f"{_API_BASE}/services/aigc/video-generation/video-synthesis"
_TASK_URL = f"{_API_BASE}/tasks/{{task_id}}"

MODEL = "wan2.7-t2v"
MAX_POLL_SECONDS = 600  # 10 min timeout per clip


class VideoGen:
    """Wan 2.7 text-to-video client for Qwen Cloud."""

    def __init__(self, qwen_api_key: str, poll_interval: float = 10.0) -> None:
        self.qwen_api_key = qwen_api_key
        self.poll_interval = poll_interval
        self._headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def run(self, storyboard: list[dict]) -> list[Path]:
        """Generate one clip per storyboard scene.

        Submits jobs sequentially to respect rate limits. Caches completed
        clips — if output/clips/scene_N.mp4 already exists it is skipped.

        Returns list of Path objects to the downloaded clip files.
        """
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []

        for entry in storyboard:
            clip_path = CLIPS_DIR / f"scene_{entry['scene']}.mp4"
            if clip_path.exists():
                clips.append(clip_path)
                continue

            prompt = entry.get("prompt", "")
            if not prompt:
                continue

            task_id = self._submit_job(prompt, entry.get("duration_seconds", 5))
            video_url = self._poll_job(task_id)
            self._download_clip(video_url, clip_path)
            clips.append(clip_path)

        return clips

    def _submit_job(self, prompt: str, duration_seconds: int) -> str:
        """POST an async video generation job. Returns the task_id."""
        body = {
            "model": MODEL,
            "input": {"prompt": prompt},
            "parameters": {
                "resolution": "720P",
                "ratio": "16:9",
                "duration": max(2, min(duration_seconds, 15)),
                "prompt_extend": True,
            },
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(_SUBMIT_URL, json=body, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in submit response: {data}")
        return task_id

    def _poll_job(self, task_id: str) -> str:
        """Poll the task endpoint until SUCCEEDED. Returns the video_url."""
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
        """Stream-download the completed MP4 to dest."""
        with httpx.stream("GET", video_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            dest.write_bytes(b"".join(r.iter_bytes()))

