"""MCP tool registration scaffold for Mars Trek WMTS."""

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config


def register_mars_trek_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register Mars Trek WMTS MCP tools."""
    pass
