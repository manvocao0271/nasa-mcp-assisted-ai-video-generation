"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_CACHE_PATH = Path.home() / ".cache" / "nasa-mcp" / "cache.sqlite3"


@dataclass(frozen=True)
class Config:
    """Resolved server configuration."""

    nasa_api_key: str
    cache_path: Path
    request_timeout: float


def load() -> Config:
    """Read config from the environment, applying defaults."""
    load_dotenv() # injects .env variables into os.environ
    nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
    if nasa_api_key == "DEMO_KEY":
        print(
            "nasa-mcp: NASA_API_KEY not set, using DEMO_KEY (w/ rate limits).",
            file=sys.stderr,
        )
    cache_path_env = os.getenv("NASA_MCP_CACHE_PATH")
    cache_path = Path(cache_path_env) if cache_path_env else DEFAULT_CACHE_PATH
    request_timeout = float(os.getenv("NASA_MCP_TIMEOUT", "30"))

    cache_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # ensure cache directory exists
    return Config(nasa_api_key, cache_path, request_timeout)
