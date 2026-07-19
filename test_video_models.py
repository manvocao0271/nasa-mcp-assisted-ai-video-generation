"""Probe every model in I2V_MODEL_PRIORITY / T2V_MODEL_PRIORITY with a
minimal real request, to confirm its schema and output actually work —
before the pipeline ever relies on it as a fallback.

Reuses VideoGen._submit_job(test_only_model=...) directly, so a PASS here
reflects the exact request body production code will send — this script
does not reimplement any schema logic of its own.

Usage:
    python test_video_models.py --all
    python test_video_models.py --models wan2.6-t2v happyhorse-1.0-r2v
    python test_video_models.py --all --image-url https://example.com/some.jpg

Notes:
- duration is fixed at 2 seconds — VideoGen's own body-builders clamp to a
  2-second floor regardless of what's requested, so 2s is the cheapest a
  real test can actually be; asking for 1 would just get silently bumped
  to 2 anyway.
- I2V/R2V/kf2v models need a reference image; T2V models don't. This script
  auto-detects which is which by checking membership in I2V_MODEL_PRIORITY
  vs T2V_MODEL_PRIORITY imported directly from video_gen.py, so it can never
  drift out of sync with the real lists.
- Each test consumes one real free-quota unit (or bills pay-as-you-go, if
  that model's Stop-on-Exhaust is off) — this is deliberate real usage, not
  a dry run. Test a handful at a time with --models rather than --all if
  you want to keep quota spend predictable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.video_gen import VideoGen, I2V_MODEL_PRIORITY, T2V_MODEL_PRIORITY
from agent.qwen_client import _load_qwen_api_keys

# apod.nasa.gov is NASA's own static archive, unchanged since the 90s —
# about as stable as a public image URL gets. Swap via --image-url if needed.
# DEFAULT_TEST_IMAGE = "https://apod.nasa.gov/apod/image/1804/SaturnRingsMoons_Cassini_967.jpg"
DEFAULT_TEST_IMAGE = "https://apod.nasa.gov/apod/image/2607/NGC474_CfhtCoelum_1500.jpg"

TEST_PROMPT = "Slow orbit around the subject, constant lighting, no camera zoom."
TEST_DURATION = 2  # seconds — see module docstring; this is the real floor
OUTPUT_DIR = Path("test_output")

# Models already verified — both schema-passing AND (where applicable)
# visually confirmed to actually reflect the reference image, not just
# "the API call succeeded." --all skips these by default; use
# --include-confirmed to force a retest (e.g. after a further code change).
CONFIRMED_MODELS = {
    "wan2.7-i2v-2026-04-25",  # schema + content confirmed
    "wan2.7-r2v-2026-06-12",  # schema + content confirmed
    "happyhorse-1.1-i2v",     # schema confirmed (post duration-floor fix)
    "happyhorse-1.0-i2v",     # schema confirmed (post duration-floor fix)
    "happyhorse-1.1-r2v",     # schema confirmed (post duration-floor fix)
    "happyhorse-1.0-r2v",     # schema confirmed (post duration-floor fix)
}


def _needs_image(model: str) -> bool:
    return model in I2V_MODEL_PRIORITY


def test_one_model(video_gen: VideoGen, model: str, image_url: str) -> tuple[bool, str]:
    """Returns (passed, message)."""
    ref_url = image_url if _needs_image(model) else ""
    try:
        t0 = time.monotonic()
        task_id = video_gen._submit_job(
            TEST_PROMPT, TEST_DURATION, ref_image_url=ref_url, test_only_model=model,
        )
        video_url = video_gen._poll_job(task_id)
        elapsed = time.monotonic() - t0

        OUTPUT_DIR.mkdir(exist_ok=True)
        dest = OUTPUT_DIR / f"{model.replace('/', '_')}.mp4"
        video_gen._download_clip(video_url, dest)
        size_kb = dest.stat().st_size / 1024

        return True, f"OK in {elapsed:.0f}s — {dest} ({size_kb:.0f} KB)"
    except Exception as exc:
        return False, str(exc)[:300]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="Specific model IDs to test")
    parser.add_argument("--all", action="store_true", help="Test every model in both priority lists (skips CONFIRMED_MODELS)")
    parser.add_argument("--include-confirmed", action="store_true", help="With --all, also retest already-confirmed models")
    parser.add_argument("--image-url", default=DEFAULT_TEST_IMAGE, help="Reference image for I2V/R2V/kf2v models")
    args = parser.parse_args()

    if not args.models and not args.all:
        parser.print_help()
        print("\nAvailable models:")
        print("  I2V:", ", ".join(I2V_MODEL_PRIORITY))
        print("  T2V:", ", ".join(T2V_MODEL_PRIORITY))
        print("\nAlready confirmed (skipped by --all unless --include-confirmed):")
        print(" ", ", ".join(sorted(CONFIRMED_MODELS)))
        sys.exit(1)

    if args.all:
        models = I2V_MODEL_PRIORITY + T2V_MODEL_PRIORITY
        if not args.include_confirmed:
            models = [m for m in models if m not in CONFIRMED_MODELS]
    else:
        models = args.models

    _qwen_keys = _load_qwen_api_keys()
    if not _qwen_keys:
        print("No Qwen Cloud API key found — set QWEN_API_KEY, or QWEN_API_KEY_0 / "
              "QWEN_API_KEY_1 / ... in .env for multi-account cycling.")
        sys.exit(1)

    video_gen = VideoGen(_qwen_keys[0], poll_interval=5.0)

    results: list[tuple[str, bool, str]] = []
    for i, model in enumerate(models, start=1):
        kind = "I2V/R2V/kf2v" if _needs_image(model) else "T2V"
        print(f"\n[{i}/{len(models)}] Testing {model} ({kind})…")
        passed, message = test_one_model(video_gen, model, args.image_url)
        print(f"  {'PASS' if passed else 'FAIL'} — {message}")
        results.append((model, passed, message))
        if i < len(models):
            time.sleep(1)  # small courtesy gap between submissions

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, p, _ in results if p)
    for model, passed, message in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {model:30s} {message}")
    print(f"\n{passed_count}/{len(results)} passed.")


if __name__ == "__main__":
    main()