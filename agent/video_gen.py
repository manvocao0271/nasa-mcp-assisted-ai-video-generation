"""Video Gen — calls Wan / HappyHorse on Qwen Cloud to generate scene clips.

For each storyboard entry, submits an image-to-video (or text-to-video) job
to the Qwen Cloud video generation API, polls until completion, downloads the
resulting clip, and saves it to output/clips/scene_N.mp4.

Wan API reference: https://www.qwencloud.com (video generation endpoint)
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

OUTPUT_DIR = Path("output")
CLIPS_DIR = OUTPUT_DIR / "clips"

# Qwen Cloud video generation endpoint (update once API docs confirmed)
QWEN_VIDEO_API_BASE = "https://api.qwencloud.com/v1/video"


class VideoGen:
    """Wan/HappyHorse video generation client for Qwen Cloud."""

    def __init__(self, qwen_api_key: str, poll_interval: float = 5.0) -> None:
        self.qwen_api_key = qwen_api_key
        self.poll_interval = poll_interval
        self.headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "Content-Type": "application/json",
        }

    def run(self, storyboard: list[dict]) -> list[Path]:
        """Generate one clip per storyboard scene.

        Submits jobs sequentially to respect rate limits. Caches completed
        clips — if output/clips/scene_N.mp4 already exists it is skipped.

        Returns list of Path objects to the downloaded clip files.
        """
        clips: list[Path] = []
        for entry in storyboard:
            clip_path = CLIPS_DIR / f"scene_{entry['scene']}.mp4"
            if clip_path.exists():
                clips.append(clip_path)
                continue

            # TODO: submit job to Qwen Cloud Wan/HappyHorse API,
            # poll until status == "completed", download video bytes,
            # write to clip_path, then append.

        return clips

    def _submit_job(self, prompt: str, ref_image_url: str, duration_seconds: int) -> str:
        """POST a video generation job. Returns the job_id."""
        # TODO: implement
        raise NotImplementedError

    def _poll_job(self, job_id: str) -> str:
        """Poll until the job is complete. Returns the video download URL."""
        # TODO: implement — poll QWEN_VIDEO_API_BASE/jobs/{job_id}
        # until status == "completed", return result.video_url
        raise NotImplementedError

    def _download_clip(self, video_url: str, dest: Path) -> None:
        """Download the completed video to dest."""
        with httpx.stream("GET", video_url) as r:
            r.raise_for_status()
            dest.write_bytes(b"".join(r.iter_bytes()))
