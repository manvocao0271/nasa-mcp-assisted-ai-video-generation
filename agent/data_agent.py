"""Data Agent — translates the user request into NASA data assets via MCP tools.

Spins up the nasa-mcp server as an MCP subprocess, hands the user message to
Qwen in tool-calling mode, runs the resulting tool calls against the MCP server,
and returns a structured assets dict saved to output/assets.json.

Output schema (output/assets.json):
    {
        "query": str,               # original user message
        "tools_called": [str],      # MCP tool names used
        "images": [                 # NASA images selected as reference frames
            {"url": str, "caption": str, "source": str}
        ],
        "data": {                   # scientific data for the script
            "<key>": <value>
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path("output")


class DataAgent:
    """Calls NASA MCP tools and collects raw assets for the pipeline."""

    def __init__(self, nasa_api_key: str) -> None:
        self.nasa_api_key = nasa_api_key

    def run(self, user_message: str) -> dict:
        """Fetch NASA data relevant to the user message.

        Returns the assets dict and writes output/assets.json.
        """
        # TODO: spin up nasa_mcp server subprocess via mcp.client.stdio,
        # pass available tools + user_message to Qwen in tool-calling mode,
        # collect tool responses, build and return assets dict.
        assets: dict = {
            "query": user_message,
            "tools_called": [],
            "images": [],
            "data": {},
        }
        (OUTPUT_DIR / "assets.json").write_text(json.dumps(assets, indent=2))
        return assets
