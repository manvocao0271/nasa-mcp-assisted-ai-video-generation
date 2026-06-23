"""Video Gen — calls Wan visual models on Qwen Cloud to generate scene clips (silent video only).

For each storyboard entry, submits an async job, polls until completion, and saves to output/clips/scene_N.mp4.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Callable

import httpx

OUTPUT_DIR = Path("output")
CLIPS_DIR = OUTPUT_DIR / "clips"

_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
_SUBMIT_URL = f"{_API_BASE}/services/aigc/video-generation/video-synthesis"
_TASK_URL = f"{_API_BASE}/tasks/{{task_id}}"

MODEL_T2V = "wan2.7-t2v"  # supports duration; swap to wan2.1-t2v-turbo for faster 5s clips
MODEL_I2V = "wan2.7-i2v"  # async, input.img, supports duration/resolution/prompt_extend
# MODEL_I2V = "happyhorse-1.0-i2v"   # sync, input.media[first_frame], duration only
# MODEL_I2V = "happyhorse-1.5-i2v"   # sync, input.media[first_frame], duration only

# Per-model I2V input format:
#   wan2.7-i2v-*     → input.img = "<url>"
#   happyhorse-*-i2v → input.media = [{"type": "first_frame", "url": "<url>"}]
# Per-model I2V parameter support:
#   wan2.7-i2v-*     → supports duration, resolution, prompt_extend
#   happyhorse-*-i2v → supports duration only
_WAN_I2V_MODELS = ("wan2.1-i2v",)  # older schema: input.img
_WAN27_I2V_MODELS = ("wan2.7-i2v", "wan2.8-i2v")  # newer schema: input.media + resolution/prompt_extend
_HAPPYHORSE_I2V_MODELS = ("happyhorse",)  # input.media, duration only
# T2V models that support the duration parameter (wan2.7+ non-turbo)
_T2V_SUPPORTS_DURATION = ("wan2.7-t2v", "wan2.8-t2v")
MAX_POLL_SECONDS = 600
MAX_SCENES = 3
CLIP_DURATION = 10  # seconds — fixed for all clips
VIDEO_RESOLUTION = "720P"



class VideoGen:
    """Wan visual-model client — one silent clip per storyboard entry."""

    MAX_DURATION = 10

    def __init__(self, qwen_api_key: str, poll_interval: float = 10.0) -> None:
        self.qwen_api_key = qwen_api_key
        self.poll_interval = poll_interval
        self.warnings: list[str] = []  # populated when I2V falls back to T2V
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
            task_id = self._submit_job(prompt, CLIP_DURATION, ref_image_url=ref_url)
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

    @staticmethod
    def _fetch_as_data_uri(url: str) -> str | None:
        """Download *url* and return a base64 data URI for inline image submission.

        For NASA image library URLs (``~large.jpg``) we first try a smaller
        ``~medium.jpg`` variant to stay within API payload limits.
        """
        _SUPPORTED = ("image/jpeg", "image/png", "image/webp")
        candidates = [url]
        if "~large." in url:
            candidates.insert(0, url.replace("~large.", "~medium."))
        for candidate in candidates:
            try:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    r = client.get(candidate)
                if r.status_code >= 400:
                    continue
                ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ct not in _SUPPORTED:
                    ct = "image/jpeg"
                b64 = base64.b64encode(r.content).decode()
                return f"data:{ct};base64,{b64}"
            except Exception:
                continue
        return None

    def _submit_job(self, prompt: str, duration_seconds: int, ref_image_url: str = "") -> str:
        raw_url = ref_image_url.strip() if ref_image_url else ""
        # Resolve the reference image to a data URI so the model backend never
        # has to fetch from hosts that block external servers (e.g. images-assets.nasa.gov).
        # Fall back to the raw URL if the download fails (e.g. non-NASA public URLs).
        if raw_url:
            media_url = self._fetch_as_data_uri(raw_url) or ""
        else:
            media_url = ""
        final_prompt = f"{prompt}\n\n{self._build_motion_directives(duration_seconds)}"

        def _i2v_body(model: str) -> dict:
            model_lower = model.lower()
            is_wan_old = any(m in model_lower for m in _WAN_I2V_MODELS)
            is_wan27 = any(m in model_lower for m in _WAN27_I2V_MODELS)
            is_happyhorse = any(m in model_lower for m in _HAPPYHORSE_I2V_MODELS)

            if is_wan_old:
                # Older wan2.1-i2v uses flat input.img field
                inp = {"prompt": final_prompt, "img": media_url}
                params: dict = {
                    "resolution": VIDEO_RESOLUTION,
                    "prompt_extend": True,
                    "duration": max(2, min(duration_seconds, 15)),
                }
            elif is_wan27:
                # wan2.7-i2v / wan2.8-i2v require input.media array
                inp = {
                    "prompt": final_prompt,
                    "media": [{"type": "first_frame", "url": media_url}],
                }
                params = {
                    "resolution": VIDEO_RESOLUTION,
                    "prompt_extend": True,
                    "duration": max(2, min(duration_seconds, 15)),
                }
            elif is_happyhorse:
                inp = {
                    "prompt": final_prompt,
                    "media": [{"type": "first_frame", "url": media_url}],
                }
                params = {"duration": max(2, min(duration_seconds, 15))}
            else:
                # Unknown I2V model — try media array format as a safe default
                inp = {
                    "prompt": final_prompt,
                    "media": [{"type": "first_frame", "url": media_url}],
                }
                params = {"duration": max(2, min(duration_seconds, 15))}

            return {"model": model, "input": inp, "parameters": params}

        def _t2v_body() -> dict:
            t2v_lower = MODEL_T2V.lower()
            supports_duration = any(m in t2v_lower for m in _T2V_SUPPORTS_DURATION)
            params: dict = {
                "resolution": VIDEO_RESOLUTION,
                "prompt_extend": True,
            }
            if supports_duration:
                params["duration"] = max(2, min(duration_seconds, 10))
            return {
                "model": MODEL_T2V,
                "input": {"prompt": final_prompt},
                "parameters": params,
            }

        def _is_quota_error(resp: httpx.Response) -> bool:
            return resp.status_code == 403 and (
                "AllocationQuota" in resp.text or "FreeTier" in resp.text
            )

        # Large base64 payloads need an extended write timeout (default 30s is too short).
        _submit_timeout = httpx.Timeout(connect=15, read=60, write=300, pool=15)
        with httpx.Client(timeout=_submit_timeout) as client:
            if media_url:
                resp = client.post(_SUBMIT_URL, json=_i2v_body(MODEL_I2V), headers=self._headers)
                if not resp.is_success:
                    err = f"{resp.status_code}: {resp.text[:200]}"
                    if _is_quota_error(resp):
                        self.warnings.append(
                            "⚠️ I2V quota exhausted. Generating without reference frame using T2V. "
                            "Check DashScope billing or try again later."
                        )
                    else:
                        self.warnings.append(
                            f"⚠️ I2V submission failed ({err}). Falling back to T2V."
                        )
                    resp = client.post(_SUBMIT_URL, json=_t2v_body(), headers=self._headers)
            else:
                resp = client.post(_SUBMIT_URL, json=_t2v_body(), headers=self._headers)

        if not resp.is_success:
            raise RuntimeError(f"Video submit failed {resp.status_code}: {resp.text}")

        task_id = resp.json().get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in submit response: {resp.json()}")
        return task_id

    def _build_motion_directives(self, duration_seconds: int) -> str:
        """Return a short block of directives forcing slow-motion orbit and constant lighting.

        These directives are purposely explicit to guide the video model:
        - Slow motion, extreme temporal deceleration — every movement is glacially slow
        - Orbit/rotate the camera around the subject at a fixed distance (no zoom/scale change)
        - Keep the entire subject fully in frame for the whole clip
        - Use smooth, continuous rotation; avoid sudden jerks
        - Maintain constant exposure/brightness for all light sources; do NOT increase bloom,
          lens flare, or brightness of stars/galaxies during the clip
        - Keep styling photorealistic with minimal dynamic color grading
        """
        return (
            "VIDEO_DIRECTIVES: Motion style — slow motion, extreme temporal deceleration, "
            "every movement glacially slow and hypnotic. "
            "Camera behavior — orbit/rotate around the subject at a fixed distance; "
            "do NOT zoom in or out (no scale change). Keep the entire subject(s) fully in frame for the entire clip. "
            "Motion: smooth, slow continuous rotation/orbit; no sudden jerks or accelerations, subtle motion-blur. "
            "Lighting: maintain constant exposure and brightness for all light sources; do NOT increase bloom, "
            "lens flares, or brightness of stars/galaxies during the duration. "
            "Style: photorealistic, cinematic, minimal color grading. "
            "Audio: no speech, no dialogue, no voiceover, no text-to-speech. "
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
        """Download the generated video clip (silent) to *dest*."""
        with httpx.stream("GET", video_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            dest.write_bytes(b"".join(r.iter_bytes()))
