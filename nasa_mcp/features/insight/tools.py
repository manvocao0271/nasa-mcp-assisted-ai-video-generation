"""MCP tool registration scaffold for InSight Mars Weather."""

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config


def register_insight_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register InSight Mars Weather MCP tools."""
    pass
