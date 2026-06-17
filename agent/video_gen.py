"""Video Gen — calls Wan visual models on Qwen Cloud to generate scene clips (silent video only).

For each storyboard entry, submits an async job, polls until completion, and saves to output/clips/scene_N.mp4.
"""

from __future__ import annotations

import base64
import math
import random
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable

import httpx

OUTPUT_DIR = Path("output")
CLIPS_DIR = OUTPUT_DIR / "clips"

_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
_SUBMIT_URL = f"{_API_BASE}/services/aigc/video-generation/video-synthesis"
_TASK_URL = f"{_API_BASE}/tasks/{{task_id}}"

MODEL_T2V = "wan2.1-t2v-turbo"
MODEL_I2V = "wan2.8-i2v-flash"          # primary I2V model
MODEL_I2V_FALLBACK = "wan2.7-i2v-2026-04-25"  # secondary I2V model
MAX_POLL_SECONDS = 600
MAX_SCENES = 3
CLIP_DURATION = 10  # seconds — fixed for all clips
VIDEO_RESOLUTION = "720P"


def _generate_ambient_wav(duration_seconds: int, out_path: Path) -> None:
    """Write a space-ambient stereo WAV to *out_path* using stdlib only.

    The result is a layered low-frequency drone (40–160 Hz) with slow LFO
    pulsing (~0.05 Hz) and a subtle noise floor — no external dependencies.
    """
    sample_rate = 44100
    n_samples = duration_seconds * sample_rate

    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(2)   # stereo
        wf.setsampwidth(2)   # 16-bit PCM
        wf.setframerate(sample_rate)

        buf = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # 2-second linear fade in/out at each end
            env = min(1.0, t / 2.0) * min(1.0, (duration_seconds - t) / 2.0)
            # Slow pulse so the drone breathes
            lfo = 0.5 + 0.5 * math.sin(2 * math.pi * 0.05 * t)

            val = (
                0.30 * math.sin(2 * math.pi * 40 * t)
                + 0.18 * math.sin(2 * math.pi * 55 * t)
                + 0.12 * math.sin(2 * math.pi * 80 * t)
                + 0.07 * math.sin(2 * math.pi * 110 * t)
                + 0.04 * math.sin(2 * math.pi * 160 * t)
                + 0.02 * (random.random() * 2 - 1)  # subtle noise floor
            )
            val = val * lfo * env
            sample = max(-32768, min(32767, int(val * 7_000)))
            packed = struct.pack("<h", sample)
            buf += packed + packed  # L == R

        wf.writeframes(bytes(buf))


def _overlay_audio_ffmpeg(
    ffmpeg: str,
    video_path: Path,
    audio_path: Path,
    out_path: Path,
) -> bool:
    """Mux *audio_path* into *video_path* → *out_path*. Returns True on success."""
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


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
        media_url = ref_image_url.strip() if ref_image_url else ""
        final_prompt = f"{prompt}\n\n{self._build_motion_directives(duration_seconds)}"

        def _i2v_body(model: str) -> dict:
            return {
                "model": model,
                "input": {"prompt": final_prompt, "img": media_url},
                "parameters": {
                    "resolution": VIDEO_RESOLUTION,
                    "prompt_extend": True,
                    "duration": max(2, min(duration_seconds, 15)),
                },
            }

        def _t2v_body() -> dict:
            return {
                "model": MODEL_T2V,
                "input": {"prompt": final_prompt},
                # T2V turbo does NOT support duration — omit the param
                "parameters": {
                    "resolution": VIDEO_RESOLUTION,
                    "prompt_extend": True,
                },
            }

        def _is_quota_error(resp: httpx.Response) -> bool:
            return resp.status_code == 403 and (
                "AllocationQuota" in resp.text or "FreeTier" in resp.text
            )

        with httpx.Client(timeout=30) as client:
            if media_url:
                # Try primary I2V model, then secondary, then fall back to T2V
                for i2v_model in (MODEL_I2V, MODEL_I2V_FALLBACK):
                    resp = client.post(_SUBMIT_URL, json=_i2v_body(i2v_model), headers=self._headers)
                    if resp.is_success:
                        break
                    err = f"{resp.status_code}: {resp.text[:200]}"
                    if not _is_quota_error(resp) and resp.status_code != 404:
                        # Hard error (bad format, auth, etc.) — surface immediately
                        raise RuntimeError(f"I2V submit failed ({i2v_model}) {err}")
                    # quota / not-found — try next model
                    self.warnings.append(f"I2V model {i2v_model} unavailable ({err}), trying next…")
                else:
                    # Both I2V models failed — fall back to T2V with a visible warning
                    self.warnings.append(
                        "⚠️ Both I2V models failed (quota/availability). "
                        "Generating without reference frame using T2V (5 s). "
                        "Check DashScope billing or try again later."
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
        """Return a short block of directives forcing ultra-slow-motion orbit and constant lighting.

        These directives are purposely explicit to guide the video model:
        - Ultra slow motion, extreme temporal deceleration — every movement is glacially slow
        - Orbit/rotate the camera around the subject at a fixed distance (no zoom/scale change)
        - Keep the entire subject fully in frame for the whole clip
        - Use smooth, continuous rotation; avoid sudden jerks
        - Maintain constant exposure/brightness for all light sources; do NOT increase bloom,
          lens flare, or brightness of stars/galaxies during the clip
        - Keep styling photorealistic with minimal dynamic color grading
        """
        return (
            "VIDEO_DIRECTIVES: Motion style — ultra slow motion, extreme temporal deceleration, "
            "every movement glacially slow and hypnotic. "
            "Camera behavior — orbit/rotate around the subject at a fixed distance; "
            "do NOT zoom in or out (no scale change). Keep the entire subject(s) fully in frame for the entire clip. "
            "Motion: smooth, ultra-slow continuous rotation/orbit; no sudden jerks or accelerations. "
            "Lighting: maintain constant exposure and brightness for all light sources; do NOT increase bloom, "
            "lens flares, or brightness of stars/galaxies during the duration. "
            "Style: photorealistic, cinematic, minimal color grading. "
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
        """Download the generated video and attempt to overlay ambient audio.

        If ffmpeg is not on PATH the clip is saved silently without error.
        """
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            raw = tmp / "raw.mp4"

            # Download
            with httpx.stream("GET", video_url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                raw.write_bytes(b"".join(r.iter_bytes()))

            # Try to add ambient audio with ffmpeg
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                audio = tmp / "ambient.wav"
                out = tmp / "with_audio.mp4"
                _generate_ambient_wav(CLIP_DURATION, audio)
                if _overlay_audio_ffmpeg(ffmpeg, raw, audio, out) and out.exists():
                    dest.write_bytes(out.read_bytes())
                    return

            # Fallback: silent video
            dest.write_bytes(raw.read_bytes())
