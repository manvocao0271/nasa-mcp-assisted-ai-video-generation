"""Space Weather Database Of Notifications, Knowledge, Information (DONKI) feature package."""

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.donki.cme import register_donki_cme_tools
from nasa_mcp.features.donki.flr import register_donki_flr_tools
from nasa_mcp.features.donki.gst import register_donki_gst_tools


def register_donki_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register all DONKI MCP tools."""

    register_donki_cme_tools(mcp, config, cache)
    register_donki_flr_tools(mcp, config, cache)
    register_donki_gst_tools(mcp, config, cache)


__all__ = ["register_donki_tools"]
