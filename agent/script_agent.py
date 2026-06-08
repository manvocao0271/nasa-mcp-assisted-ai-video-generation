"""Script Agent — writes a 3-act narration script grounded in NASA assets.

Reads output/assets.json, calls Qwen to produce a structured screenplay,
and writes output/script.md.

Output schema (script dict):
    {
        "title": str,
        "scenes": [
            {
                "act": int,             # 1 | 2 | 3
                "narration": str,       # spoken narration (~100 words)
                "mood": str,            # cinematic mood descriptor
                "data_ref": str         # which assets.data key this scene draws from
            }
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path("output")

SYSTEM_PROMPT = """\
You are a science documentary scriptwriter. Write a 3-act short-film script (~300 words total)
grounded strictly in the NASA data provided. Each act is one scene. Be cinematic and factual.
Return JSON matching this schema exactly:
{
  "title": "<episode title>",
  "scenes": [
    {"act": 1, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"},
    {"act": 2, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"},
    {"act": 3, "narration": "...", "mood": "...", "data_ref": "<key from assets.data>"}
  ]
}
"""


class ScriptAgent:
    """Generates a 3-act narration script from NASA assets using Qwen."""

    def __init__(self, qwen_api_key: str) -> None:
        self.qwen_api_key = qwen_api_key

    def run(self, assets: dict, user_message: str) -> dict:
        """Generate script from assets. Returns script dict and writes output/script.md.

        Args:
            assets: output from DataAgent.run()
            user_message: original user request for tone/context
        """
        # TODO: call Qwen chat API with SYSTEM_PROMPT + assets JSON as user message,
        # parse JSON response into script dict.
        script: dict = {
            "title": "Untitled Episode",
            "scenes": [
                {"act": 1, "narration": "", "mood": "epic", "data_ref": ""},
                {"act": 2, "narration": "", "mood": "tense", "data_ref": ""},
                {"act": 3, "narration": "", "mood": "reflective", "data_ref": ""},
            ],
        }

        md_lines = [f"# {script['title']}\n"]
        for scene in script["scenes"]:
            md_lines.append(f"## Act {scene['act']} — {scene['mood']}\n")
            md_lines.append(scene["narration"] + "\n")
        (OUTPUT_DIR / "script.md").write_text("\n".join(md_lines))

        return script
