"""Video Gen — calls Wan visual models on Qwen Cloud to generate scene clips (silent video only).

For each storyboard entry, submits an async job, polls until completion, and saves to output/clips/scene_N.mp4.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

from agent.qwen_client import _load_qwen_api_keys, _log_api_call

_DEFAULT_OUTPUT_DIR = Path("output")
_DEFAULT_CLIPS_DIR = _DEFAULT_OUTPUT_DIR / "clips"

_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
_SUBMIT_URL = f"{_API_BASE}/services/aigc/video-generation/video-synthesis"
_TASK_URL = f"{_API_BASE}/tasks/{{task_id}}"

# Pinned to a dated snapshot for the same reason as MODEL_I2V above — the
# bare "wan2.7-t2v" alias may draw from a different quota bucket.
MODEL_T2V = "wan2.7-t2v-2026-06-12"  # supports duration; swap to wan2.1-t2v-turbo for faster 5s clips

# Pinned to the exact dated snapshot shown in the DashScope console, not the
# bare "wan2.7-i2v" alias — free-tier quota is tracked per exact model
# string, and the alias can draw from a different (already-exhausted) bucket
# than the dated snapshot even though they're "the same model".
MODEL_I2V = "wan2.7-i2v-2026-04-25"

# Tried in order on every I2V request; on quota exhaustion (or any other
# submission failure) for one model, the next one is tried automatically.
# Falls through to MODEL_T2V only once every candidate here has failed.
# Ordered by remaining free-tier quota (highest first) among model families
# _submit_job already knows how to format a request body for — see
# _WAN_I2V_MODELS / _WAN27_I2V_MODELS / _HAPPYHORSE_I2V_MODELS below.
# Update this list's order/contents as your DashScope quota page changes.
I2V_MODEL_PRIORITY = [
    MODEL_I2V,               # wan2.7-i2v-2026-04-25 — current default, best quality/prompt fit
    "wan2.1-i2v-plus",       # 200 remaining — older gen, input.img_url schema, high quality
    "wan2.1-i2v-turbo",      # 200 remaining — older gen, faster/lighter
    "wan2.7-r2v-2026-06-12", # confirmed schema (see _R2V_MODELS below) — used single-reference,
                              # silent-only; its multi-reference/voice-cloning capability is
                              # intentionally unused here, doesn't fit this app's silent clips
    "happyhorse-1.1-i2v",    # 10 remaining
    "happyhorse-1.0-i2v",    # 10 remaining
    "happyhorse-1.1-r2v",    # 10 remaining — same single-reference treatment as wan2.7-r2v above
    "happyhorse-1.0-r2v",    # 10 remaining — schema inferred from the happyhorse-1.1-r2v example
                              # (same doc page covered both versions, not independently confirmed)
]
# async, input.img_url, supports duration/resolution/prompt_extend
# MODEL_I2V = "happyhorse-1.5-i2v"   # sync, input.media[first_frame], duration only

# Per-model I2V input format:
#   wan2.7-i2v-*     → input.img_url = "<url>" (confirmed via live test — the DashScope docs
#                       comment this was originally written from called it "img", which is wrong)
#   happyhorse-*-i2v → input.media = [{"type": "first_frame", "url": "<url>"}]
# Per-model R2V input format (confirmed via DashScope example curls):
#   wan2.7-r2v-*, happyhorse-*-r2v → input.media = [{"type": "reference_image", "url": "<url>"}]
#   Real R2V supports multiple media items (reference_image AND reference_video, each with an
#   optional reference_voice for dialogue/voice cloning) — deliberately not used here; this app
#   only ever sends one silent reference_image, since its prompts are motion-directives with
#   explicitly no dialogue, not the indexed "[Image 1]/[Video 1]" multi-character scripts R2V is
#   really designed for.
# wan2.1-kf2v-plus was tried and removed — confirmed via live test that it requires BOTH
# first_frame_url and last_frame_url (omitting the second returns a URL validation error), which
# doesn't fit this app's single-reference-image architecture. Not worth a same-frame-twice
# workaround without separately verifying that actually produces usable output.
# Per-model parameter support:
#   wan2.7-i2v-*, wan2.7-r2v-* → duration, resolution, prompt_extend
#   happyhorse-*-i2v/r2v/t2v  → duration only (confirmed minimum is 3, not 2) — r2v also takes
#                                 resolution + ratio (no prompt_extend — unconfirmed whether it's
#                                 silently ignored or rejected, so omitted)
_WAN_I2V_MODELS = ("wan2.1-i2v",)  # older schema: input.img_url
_WAN27_I2V_MODELS = ("wan2.7-i2v", "wan2.8-i2v")  # newer schema: input.media + resolution/prompt_extend
_HAPPYHORSE_I2V_MODELS = ("happyhorse",)  # input.media, duration only — matches happyhorse-*-i2v
_R2V_MODELS = ("wan2.7-r2v", "happyhorse-1.1-r2v", "happyhorse-1.0-r2v")  # input.media[reference_image]
# T2V models confirmed to support the duration parameter. Models NOT listed
# here still work as fallbacks, just without an explicit duration — DashScope
# uses that model's own fixed default instead (e.g. wan2.1-t2v-turbo is ~5s),
# so a clip generated via one of those may run shorter than the rest of the
# episode. Verify and add a model here once you've confirmed it accepts
# "duration" rather than guessing, since a wrong guess is a silent 400, not
# a loud one.
_T2V_SUPPORTS_DURATION = ("wan2.7-t2v", "wan2.8-t2v")

# Tried in order once every I2V candidate above has failed — this is the
# genuine last line of defense; if every model here is also exhausted,
# _submit_job raises for real and the pipeline reports a hard failure (there
# is no tier below T2V to fall back to further).
T2V_MODEL_PRIORITY = [
    MODEL_T2V,                # wan2.7-t2v-2026-06-12 — current default
    "wan2.7-t2v-2026-04-25",
    "wan2.6-t2v",
    "wan2.2-t2v-plus",
    "wan2.5-t2v-preview",
    "wan2.1-t2v-plus",        # 200 remaining, confirmed duration support
    "happyhorse-1.1-t2v",     # 10 remaining
    "wan2.1-t2v-turbo",       # 200 remaining, LAST — no duration support (~5s fixed,
                               # will run shorter than the rest of the episode)
]
MAX_POLL_SECONDS = 600
MAX_SCENES = 3
CLIP_DURATION = 10  # seconds — fixed for all clips
VIDEO_RESOLUTION = "720P"



class VideoGen:
    """Wan visual-model client — one silent clip per storyboard entry."""

    MAX_DURATION = 10

    def __init__(
        self,
        qwen_api_key: str,
        poll_interval: float = 10.0,
        cancel_event: threading.Event | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.qwen_api_key = qwen_api_key  # kept for backward compat / display
        self._api_keys = _load_qwen_api_keys(fallback=qwen_api_key)
        if not self._api_keys:
            raise ValueError(
                "No Qwen Cloud API key available for video generation. Set "
                "QWEN_API_KEY, or QWEN_API_KEY_0 / QWEN_API_KEY_1 / ... in "
                ".env for multi-account cycling."
            )
        self.poll_interval = poll_interval
        self.warnings: list[str] = []  # populated when I2V falls back to T2V
        self.last_model_used: str = ""  # which model the most recent _submit_job used
        # Which key actually succeeded on the most recent _submit_job call —
        # _poll_job MUST use this same key, not just any configured key: a
        # task_id is scoped to the account that submitted it, so polling
        # with a different account's key would fail even though that key
        # is otherwise perfectly valid.
        self._last_used_key: str = self._api_keys[0]
        self.cancel_event = cancel_event
        _out = output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR
        self.clips_dir = _out / "clips"

    @staticmethod
    def _headers_for(key: str) -> dict:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def run(
        self,
        storyboard: list[dict],
        on_clip_start: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Generate one clip per storyboard entry (up to MAX_SCENES)."""
        self.clips_dir.mkdir(parents=True, exist_ok=True)
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

    def _unique_clip_path(self, scene: int) -> Path:
        candidate = self.clips_dir / f"scene_{scene}.mp4"
        counter = 1
        while candidate.exists():
            candidate = self.clips_dir / f"scene_{scene}_{counter}.mp4"
            counter += 1
        return candidate

    # Same rationale/constants as ScriptAgent/StoryboardAgent's identical
    # resize step — kept in sync across all three since they all embed the
    # same kind of reference image.
    _MAX_IMAGE_DIMENSION = 1568
    _RESIZE_SKIP_THRESHOLD_BYTES = 1_500_000

    @classmethod
    def _fetch_as_data_uri(cls, url: str) -> str | None:
        """Download *url* and return a base64 data URI for inline image submission.

        For NASA image library URLs (``~large.jpg``) we first try a smaller
        ``~medium.jpg`` variant to stay within API payload limits. Large
        downloads (e.g. full-resolution APOD originals, which can run
        20-30MB+ with no smaller variant available via URL substitution)
        are resized before encoding — Wan's own server-side fetcher has been
        observed failing outright ("Failed to download <url>") on files this
        large, the same failure mode we already fixed client-side for the
        vision-model calls in ScriptAgent/StoryboardAgent.
        """
        _SUPPORTED = ("image/jpeg", "image/png", "image/webp")
        # NASA Image Library only allows ~thumb.jpg from external servers;
        # ~large.jpg and ~medium.jpg return 403.  Try the smallest accessible
        # variant first, then progressively larger, then the original URL.
        candidates: list[str] = []
        if "~large." in url:
            candidates.append(url.replace("~large.", "~thumb."))
            candidates.append(url.replace("~large.", "~medium."))
        elif "~orig." in url:
            candidates.append(url.replace("~orig.", "~thumb."))
            candidates.append(url.replace("~orig.", "~medium."))
        candidates.append(url)
        for candidate in candidates:
            try:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    r = client.get(candidate)
                if r.status_code >= 400:
                    continue
                ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ct not in _SUPPORTED:
                    ct = "image/jpeg"

                raw = r.content
                if len(raw) > cls._RESIZE_SKIP_THRESHOLD_BYTES:
                    try:
                        import io
                        from PIL import Image
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        img.thumbnail(
                            (cls._MAX_IMAGE_DIMENSION, cls._MAX_IMAGE_DIMENSION),
                            Image.Resampling.LANCZOS,
                        )
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        raw = buf.getvalue()
                        ct = "image/jpeg"
                    except Exception:
                        pass  # fall through and try sending the original

                b64 = base64.b64encode(raw).decode()
                return f"data:{ct};base64,{b64}"
            except Exception:
                continue
        return None

    def _submit_job(
        self,
        prompt: str,
        duration_seconds: int,
        ref_image_url: str = "",
        test_only_model: str | None = None,
    ) -> str:
        """Submit a generation job.

        test_only_model: diagnostic-only. When set, submits to exactly this
        model — no priority-list looping, no fallback to T2V on failure,
        raises immediately if it fails. Reuses the same _i2v_body/_r2v_body/
        _t2v_body closures as the real production path below, so
        a passing test actually reflects what ships — it isn't a separate
        reimplementation that could quietly drift out of sync.
        """
        raw_url = ref_image_url.strip() if ref_image_url else ""

        # Always try to pre-fetch (and resize if needed) ourselves rather
        # than passing the raw URL straight through and trusting Wan's
        # server-side fetcher to succeed — that's failed outright on large
        # NASA originals from domains other than images-assets.nasa.gov
        # (e.g. apod.nasa.gov), so this is no longer domain-specific. Only
        # fall back to the raw URL if our own fetch attempt fails, since a
        # working raw URL is still better than no reference image at all.
        media_url: str = (self._fetch_as_data_uri(raw_url) or raw_url) if raw_url else ""

        final_prompt = f"{prompt}\n\n{self._build_motion_directives(duration_seconds)}"

        def _i2v_body(model: str) -> dict:
            model_lower = model.lower()
            is_wan_old = any(m in model_lower for m in _WAN_I2V_MODELS)
            is_wan27 = any(m in model_lower for m in _WAN27_I2V_MODELS)
            is_happyhorse = any(m in model_lower for m in _HAPPYHORSE_I2V_MODELS)

            if is_wan_old:
                # Confirmed via live test: field is img_url, not img — API
                # returned "img_url must be set for image to video" otherwise.
                inp = {"prompt": final_prompt, "img_url": media_url}
                # Confirmed via live test: duration must be exactly 3, 4, or
                # 5 — a discrete set, not a continuous range like wan2.7's.
                # max(3, min(x, 5)) lands on exactly one of those three for
                # any integer input, so this app's 10s target always clamps
                # to 5 here — this model can never produce a full-length clip.
                params: dict = {
                    "resolution": VIDEO_RESOLUTION,
                    "prompt_extend": True,
                    "duration": max(3, min(duration_seconds, 5)),
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
                # Confirmed via live test: happyhorse's minimum duration is
                # 3, not 2 — API returned "duration must be between 3 and 15".
                params = {"duration": max(3, min(duration_seconds, 15))}
            else:
                # Unknown I2V model — try media array format as a safe default
                inp = {
                    "prompt": final_prompt,
                    "media": [{"type": "first_frame", "url": media_url}],
                }
                params = {"duration": max(2, min(duration_seconds, 15))}

            return {"model": model, "input": inp, "parameters": params}

        def _r2v_body(model: str) -> dict:
            """R2V, restricted to a single silent reference_image — see _R2V_MODELS
            comment above for why the multi-reference/voice-cloning capability this
            schema actually supports is deliberately not used.

            Schema confirmed from DashScope example curls for wan2.7-r2v-2026-06-12
            and happyhorse-1.1-r2v/happyhorse-1.0-r2v. "ratio" is new here — no other
            model family in this file uses it. happyhorse's example omitted
            prompt_extend/watermark entirely, so they're left off for that family
            rather than guessing whether it accepts and ignores them or rejects them.
            """
            model_lower = model.lower()
            is_happyhorse = any(m in model_lower for m in _HAPPYHORSE_I2V_MODELS)

            inp = {
                "prompt": final_prompt,
                "media": [{"type": "reference_image", "url": media_url}],
            }
            if is_happyhorse:
                # Confirmed via live test on happyhorse i2v (same family,
                # same minimum applies): duration floor is 3, not 2.
                params: dict = {
                    "resolution": VIDEO_RESOLUTION,
                    "ratio": "16:9",
                    "duration": max(3, min(duration_seconds, 15)),
                }
            else:
                # wan2.7-r2v family. The example curl sets prompt_extend:false and
                # watermark:true — deliberately overridden here to match this app's
                # existing wan2.7 convention (prompt_extend on) and to avoid a
                # visible watermark on generated clips, rather than copying the
                # example's choices verbatim.
                params = {
                    "resolution": VIDEO_RESOLUTION,
                    "ratio": "16:9",
                    "prompt_extend": True,
                    "watermark": False,
                    "duration": max(2, min(duration_seconds, 15)),
                }
            return {"model": model, "input": inp, "parameters": params}

        def _build_i2v_or_r2v_body(model: str) -> dict:
            """Dispatch to the R2V schema builder for R2V models, I2V schema
            otherwise — checked first since "happyhorse" alone
            (_HAPPYHORSE_I2V_MODELS) would otherwise misclassify
            happyhorse-*-r2v as a plain I2V model."""
            model_lower = model.lower()
            if any(m in model_lower for m in _R2V_MODELS):
                return _r2v_body(model)
            return _i2v_body(model)

        def _t2v_body(model: str) -> dict:
            model_lower = model.lower()
            supports_duration = any(m in model_lower for m in _T2V_SUPPORTS_DURATION)
            is_happyhorse = any(m in model_lower for m in _HAPPYHORSE_I2V_MODELS)

            if is_happyhorse:
                # Confirmed via live test: happyhorse's duration floor is 3,
                # not 2 — API returned "duration must be between 3 and 15".
                params: dict = {"duration": max(3, min(duration_seconds, 15))}
            else:
                params = {"resolution": VIDEO_RESOLUTION, "prompt_extend": True}
                if supports_duration:
                    params["duration"] = max(2, min(duration_seconds, 10))

            return {
                "model": model,
                "input": {"prompt": final_prompt},
                "parameters": params,
            }

        def _is_quota_error(resp: httpx.Response) -> bool:
            return resp.status_code == 403 and (
                "AllocationQuota" in resp.text or "FreeTier" in resp.text
            )

        def _extract_request_id(resp: httpx.Response) -> str:
            """DashScope includes a top-level request_id in both success and
            error response bodies — used for terminal logging, matching
            the Request ID column in DashScope's own console log."""
            try:
                return resp.json().get("request_id", "") or ""
            except Exception:
                return ""

        def _submit_t2v_chain(client: httpx.Client) -> httpx.Response:
            """Try every T2V candidate in order; for each model, cycle
            through every configured API key (separate accounts have
            separate free-tier quota pools) before moving to the next
            model. Last attempt raises for real."""
            if not T2V_MODEL_PRIORITY:
                raise RuntimeError("T2V_MODEL_PRIORITY is empty — no T2V model to fall back to.")

            _resp: httpx.Response | None = None
            for _t2v_model in T2V_MODEL_PRIORITY:
                _body = _t2v_body(_t2v_model)
                for _key_idx, _key in enumerate(self._api_keys):
                    _t0 = time.monotonic()
                    _resp = client.post(_SUBMIT_URL, json=_body, headers=self._headers_for(_key))
                    _latency = time.monotonic() - _t0
                    _log_api_call(_key, _t2v_model, _latency, str(_resp.status_code), _extract_request_id(_resp))
                    if _resp.is_success:
                        self.last_model_used = _t2v_model
                        self._last_used_key = _key
                        print(f"[VideoGen] T2V submitted on {_t2v_model} using key #{_key_idx}")
                        return _resp
                    _err = f"{_resp.status_code}: {_resp.text[:200]}"
                    if _is_quota_error(_resp):
                        print(f"[VideoGen] {_t2v_model} quota exhausted on key #{_key_idx} — trying next key")
                        continue
                    # Non-quota error (bad params, etc.) — identical request
                    # would fail the same way on every other key too, so
                    # don't waste attempts cycling keys; move to next model.
                    print(f"[VideoGen] {_t2v_model} submission failed ({_err}) — non-quota error, moving to next model")
                    break
            # Every T2V candidate failed on every configured key too — this
            # is a genuine hard failure, there is no further fallback tier.
            self.warnings.append(
                f"⚠️ All T2V models exhausted or failed across all {len(self._api_keys)} "
                "configured API key(s) — no fallback left. Check DashScope billing, "
                "or add more models to T2V_MODEL_PRIORITY."
            )
            assert _resp is not None  # loop ran ≥1 time since the list isn't empty
            return _resp

        # Large base64 payloads need an extended write timeout (default 30s is too short).
        _submit_timeout = httpx.Timeout(connect=15, read=60, write=300, pool=15)
        with httpx.Client(timeout=_submit_timeout) as client:
            if test_only_model:
                # Diagnostic path — exactly one model, first configured key,
                # no fallback, raises on failure instead of trying anything else.
                _body = _build_i2v_or_r2v_body(test_only_model) if media_url else _t2v_body(test_only_model)
                _key = self._api_keys[0]
                _t0 = time.monotonic()
                resp = client.post(_SUBMIT_URL, json=_body, headers=self._headers_for(_key))
                _log_api_call(_key, test_only_model, time.monotonic() - _t0, str(resp.status_code), _extract_request_id(resp))
                if not resp.is_success:
                    raise RuntimeError(f"[TEST] {test_only_model} submit failed {resp.status_code}: {resp.text}")
                self.last_model_used = test_only_model
                self._last_used_key = _key
                task_id = resp.json().get("output", {}).get("task_id")
                if not task_id:
                    raise RuntimeError(f"[TEST] {test_only_model} — no task_id in response: {resp.json()}")
                return task_id

            if media_url:
                _uri_type = "data URI" if media_url.startswith("data:") else "URL"
                print(f"[VideoGen] I2V mode — reference image as {_uri_type} ({len(media_url)} chars)")

                resp: httpx.Response | None = None
                _i2v_success = False
                for _model in I2V_MODEL_PRIORITY:
                    _body = _build_i2v_or_r2v_body(_model)
                    for _key_idx, _key in enumerate(self._api_keys):
                        _t0 = time.monotonic()
                        resp = client.post(_SUBMIT_URL, json=_body, headers=self._headers_for(_key))
                        _log_api_call(_key, _model, time.monotonic() - _t0, str(resp.status_code), _extract_request_id(resp))
                        if resp.is_success:
                            self.last_model_used = _model
                            self._last_used_key = _key
                            print(f"[VideoGen] I2V submitted on {_model} using key #{_key_idx}")
                            _i2v_success = True
                            break
                        err = f"{resp.status_code}: {resp.text[:200]}"
                        if _is_quota_error(resp):
                            print(f"[VideoGen] {_model} quota exhausted on key #{_key_idx} — trying next key")
                            continue
                        # Non-quota error — identical request would fail the
                        # same way on every other key, so don't waste
                        # attempts cycling keys; move to the next model.
                        print(f"[VideoGen] {_model} submission failed ({err}) — non-quota error, moving to next model")
                        break
                    if _i2v_success:
                        break

                if not _i2v_success:
                    # Every I2V (model, key) combination failed — fall back to the T2V chain.
                    self.warnings.append(
                        f"⚠️ All I2V models exhausted or failed across all {len(self._api_keys)} "
                        "configured API key(s). Generating without reference frame using T2V."
                    )
                    print("[VideoGen] All I2V candidates failed on all keys, falling back to T2V chain")
                    resp = _submit_t2v_chain(client)
            else:
                print(f"[VideoGen] T2V mode — no reference image (ref_image_url empty or fetch failed)")
                resp = _submit_t2v_chain(client)

        # resp is always reassigned from None by this point on every actual
        # code path (either a successful I2V/T2V submission, or the T2V
        # chain fallback — which itself always returns a real Response, see
        # its own assert) — but the type checker can't prove that across the
        # nested loops above, so narrow explicitly. Same pattern as
        # _submit_t2v_chain's own assert just above.
        assert resp is not None
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
        # Must use the same key that submitted this task — a task_id is
        # scoped to the account that created it, not shared across accounts.
        poll_headers = {"Authorization": f"Bearer {self._last_used_key}"}
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

            # Interruptible wait: wakes immediately if cancel_event is set
            if self.cancel_event:
                if self.cancel_event.wait(self.poll_interval):
                    raise InterruptedError(f"Video generation cancelled (task={task_id})")
            else:
                time.sleep(self.poll_interval)

        raise TimeoutError(f"Video generation timed out after {MAX_POLL_SECONDS}s (task={task_id})")

    def _download_clip(self, video_url: str, dest: Path) -> None:
        """Download the generated video clip (silent) to *dest*."""
        with httpx.stream("GET", video_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            dest.write_bytes(b"".join(r.iter_bytes()))