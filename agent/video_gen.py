"""Video Gen — calls Wan 2.7 on Qwen Cloud to generate scene clips.

For each storyboard entry, submits an async text-to-video job to the DashScope international API, polls until completion, downloads the MP4, and saves it to output/clips/scene_N.mp4.

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

MODEL_T2V = "happyhorse-1.0-t2v"      # text-to-video (no ref image)
MODEL_I2V = "wan2.7-i2v-2026-04-25"  # image-to-video (NASA first frame → animated clip)
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

    MAX_CLIPS = 1
    MAX_DURATION = 10  # seconds

    def run(self, storyboard: list[dict]) -> list[Path]:
        """Generate up to MAX_CLIPS new clips from the storyboard.

        Always writes a new uniquely-named file so clips accumulate across runs.

        Returns list of Path objects to the downloaded clip files.
        """
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []

        for entry in storyboard[: self.MAX_CLIPS]:
            prompt = entry.get("prompt", "")
            if not prompt:
                continue

            ref_url = entry.get("ref_image_url", "")
            # Pick a unique filename so clips accumulate rather than overwrite
            clip_path = self._unique_clip_path(entry["scene"])
            duration = min(entry.get("duration_seconds", 5), self.MAX_DURATION)
            task_id = self._submit_job(prompt, duration, ref_image_url=ref_url)
            video_url = self._poll_job(task_id)
            self._download_clip(video_url, clip_path)
            clips.append(clip_path)

        return clips

    @staticmethod
    def _unique_clip_path(scene: int) -> Path:
        """Return output/clips/scene_N.mp4, incrementing N until the name is free."""
        candidate = CLIPS_DIR / f"scene_{scene}.mp4"
        counter = 1
        while candidate.exists():
            candidate = CLIPS_DIR / f"scene_{scene}_{counter}.mp4"
            counter += 1
        return candidate

    def _submit_job(self, prompt: str, duration_seconds: int, ref_image_url: str = "") -> str:
        """POST an async video generation job. Returns the task_id.

        Uses wan2.7-i2v (image-to-video) when a valid NASA ref image URL is provided AND it is visually relevant to the prompt — the image becomes the first frame, giving Wan accurate visual grounding. Falls back to wan2.7-t2v (text-to-video) otherwise.
        """
        use_ref = (
            ref_image_url
            and self._url_is_usable_image(ref_image_url)
            and self._ref_matches_prompt(ref_image_url, prompt)
        )
        if use_ref:
            model = MODEL_I2V
            inp: dict = {
                "prompt": prompt,
                "media": [{"type": "first_frame", "url": ref_image_url}],
            }
        else:
            model = MODEL_T2V
            inp = {"prompt": prompt}

        body = {
            "model": model,
            "input": inp,
            "parameters": {
                "resolution": "720P",
                "duration": max(2, min(duration_seconds, 15)),
                "prompt_extend": True,
            },
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

    @staticmethod
    def _ref_matches_prompt(ref_url: str, prompt: str) -> bool:
        """Heuristic: reject an APOD ref image that clearly doesn't match the prompt topic.

        APOD URLs contain the image filename which often hints at the subject. If the prompt is about Mars/Earth/Moon/asteroid but the URL suggests a galaxy, nebula, or unrelated object, skip the ref so t2v is used.
        """
        prompt_lower = prompt.lower()
        url_lower = ref_url.lower()

        # Topic keywords the prompt is about
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
                return False  # prompt expects this topic but URL doesn't show it

        return True

    @staticmethod
    def _url_is_usable_image(url: str) -> bool:
        """Check that the URL serves a supported image format AND is reachable by Wan's servers.

        Some NASA hosts (images-assets.nasa.gov) are inaccessible from Wan's remote download infrastructure even though they respond to local HEAD requests.  These domains are blocklisted so we fall back to t2v.
        """
        _SUPPORTED = ("image/jpeg", "image/png", "image/gif", "image/webp")
        # Domains confirmed unreachable by Wan's video generation servers
        _WAN_BLOCKED_DOMAINS = (
            "images-assets.nasa.gov",
        )
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

