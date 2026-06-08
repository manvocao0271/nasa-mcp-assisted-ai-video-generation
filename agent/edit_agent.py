"""Edit Agent — assembles scene clips into the final film using ffmpeg.

Takes the ordered list of clip paths and the script (for captions), then:
  1. Concatenates clips in scene order
  2. Burns in narration captions from script.md
  3. Applies cross-fade transitions between scenes
  4. Writes output/episode_final.mp4

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

OUTPUT_DIR = Path("output")
FINAL_VIDEO = OUTPUT_DIR / "episode_final.mp4"
FADE_DURATION = 0.5  # seconds


class EditAgent:
    """ffmpeg-based video assembly: concat + captions + fades."""

    def run(self, clips: list[Path], script: dict) -> Path:
        """Assemble clips into a single film with captions.

        Args:
            clips: ordered list of scene clip Paths from VideoGen.run()
            script: script dict from ScriptAgent.run()

        Returns Path to the final assembled video.
        """
        if not clips:
            raise ValueError("No clips provided to EditAgent")

        concat_path = self._concat_clips(clips)
        final_path = self._burn_captions(concat_path, script)
        return final_path

    def _concat_clips(self, clips: list[Path]) -> Path:
        """Concatenate clips using ffmpeg concat demuxer with crossfade.

        Writes a temporary concat list file, runs ffmpeg, returns Path to
        the concatenated (uncaptioned) video.
        """
        concat_list = OUTPUT_DIR / "concat_list.txt"
        concat_list.write_text(
            "\n".join(f"file '{clip.resolve()}'" for clip in clips)
        )
        concat_output = OUTPUT_DIR / "concat.mp4"

        # TODO: replace with xfade filter for crossfade transitions between clips
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(concat_output),
            ],
            check=True,
        )
        return concat_output

    def _burn_captions(self, video_path: Path, script: dict) -> Path:
        """Burn narration text as subtitles into the video using ffmpeg drawtext.

        Writes output/episode_final.mp4.
        """
        # TODO: generate an SRT/ASS subtitle file from script scenes + timing,
        # then use ffmpeg subtitles filter to burn them in.
        # For now, copy the concat output as the final video.
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-c", "copy",
                str(FINAL_VIDEO),
            ],
            check=True,
        )
        return FINAL_VIDEO
