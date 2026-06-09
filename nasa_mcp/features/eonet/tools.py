"""MCP tool registration scaffold for EONET."""

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config


def register_eonet_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register EONET MCP tools."""
    pass
