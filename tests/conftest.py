"""Shared fixtures for root-level integration tests."""

from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.apod.tools import register_apod_tools
from nasa_mcp.features.earth.tools import register_earth_tools
from nasa_mcp.features.mars_rovers.tools import register_mars_rover_tools
from nasa_mcp.features.neo.tools import register_neo_tools


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    """A SQLite cache path inside pytest's tmp_path. Auto-cleaned after the test."""
    return tmp_path / "test_cache.sqlite3"


@pytest.fixture
def test_config(tmp_cache_path: Path) -> Config:
    """A Config pointing at a throwaway cache and a dummy API key."""
    return Config(
        nasa_api_key="TEST_KEY",
        cache_path=tmp_cache_path,
        request_timeout=5.0,
    )


@pytest.fixture
def cache(tmp_cache_path: Path) -> Cache:
    """A fresh Cache backed by a tmp file."""
    return Cache(tmp_cache_path)


@pytest.fixture
def mcp_with_tools(test_config: Config, cache: Cache) -> FastMCP:
    """A FastMCP instance with all feature tools registered."""
    mcp = FastMCP("nasa-mcp-test")
    register_apod_tools(mcp, test_config, cache)
    register_earth_tools(mcp, test_config, cache)
    register_mars_rover_tools(mcp, test_config, cache)
    register_neo_tools(mcp, test_config, cache)
    return mcp
