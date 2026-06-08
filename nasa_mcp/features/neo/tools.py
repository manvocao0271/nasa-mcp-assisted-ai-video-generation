"""MCP tool registration for NeoWs."""

from datetime import date, timedelta
import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.neo.api import get_neo_feed, get_neo_lookup
from nasa_mcp.features.neo.inputs import GetNeoFeedInput, GetNeoLookupInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600


def register_neo_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register NeoWs MCP tools."""

    @mcp.tool()
    async def get_neo_feed_tool(args: GetNeoFeedInput):
        """Fetch all asteroids making a close approach to Earth within a date range (max 7 days).

        Returns a dict with `element_count` (total asteroids) and `near_earth_objects`, a dict keyed by date string (e.g. `"2024-01-15"`), each containing a list of asteroid objects. Each asteroid includes `id`, `name`, `is_potentially_hazardous_asteroid`, `estimated_diameter` (in km/m/ft/miles), and `close_approach_data` with `close_approach_date`, `relative_velocity` (km/s and mph), and `miss_distance` (km, lunar, and astronomical units).

        Use this for questions like "which asteroids are passing Earth this week", "are any hazardous asteroids approaching between Jan 10 and Jan 15 2025", or "how close did asteroids come to Earth on March 3 2024".
        """

        key_input = f"{args.start_date}|{args.end_date}"
        key = f"neo:get_neo_feed:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached
        
        response = await get_neo_feed(config, args.start_date, args.end_date)

        end = args.end_date or (args.start_date + timedelta(days=7))
        ttl_seconds = SHORT_TTL if end >= date.today() else LONG_TTL

        cache.set(key, response, ttl_seconds=ttl_seconds)

        return response
    
    @mcp.tool()
    async def get_neo_lookup_tool(args: GetNeoLookupInput):
        """Fetch complete data for a single asteroid by its SPK-ID.

        Returns a dict with `id`, `name`, `designation`, `absolute_magnitude_h`, `is_potentially_hazardous_asteroid`, `estimated_diameter` (km/m/ft/miles), `close_approach_data` (full historical and future approach list), and `orbital_data` (epoch, semi-major axis, eccentricity, inclination, etc.).

        Use this when you already have a specific asteroid's SPK-ID — for example from a prior `get_neo_feed_tool` result — and need its full orbital profile or complete approach history. Example: SPK-ID `"3542519"` returns data for asteroid (2010 PK9).
        """

        key = f"neo:get_neo_lookup:{hashlib.sha256(str(args.asteroid_id).encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached
        
        response = await get_neo_lookup(config, args.asteroid_id)

        ttl_seconds = LONG_TTL

        cache.set(key, response, ttl_seconds=ttl_seconds)

        return response