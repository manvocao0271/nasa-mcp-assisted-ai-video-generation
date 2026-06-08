"""MCP server entry point. Registers tools and runs the stdio transport."""

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import load
from nasa_mcp.features.apod.tools import register_apod_tools
from nasa_mcp.features.earth.tools import register_earth_tools
# from nasa_mcp.features.mars_rovers.tools import register_mars_rover_tools # deprecated
from nasa_mcp.features.neo.tools import register_neo_tools

mcp = FastMCP("nasa-mcp")
config = load()
cache = Cache(config.cache_path)


def register_tools() -> None:
    """Register all MCP feature modules."""
    
    register_apod_tools(mcp, config, cache)
    register_earth_tools(mcp, config, cache)
    # register_mars_rover_tools(mcp, config, cache)
    register_neo_tools(mcp, config, cache)


register_tools()


def main() -> None:
    """Run the nasa-mcp server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
