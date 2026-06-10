"""MCP tool registration for FLR."""

from datetime import date
import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.donki.flr.api import get_flr_events
from nasa_mcp.features.donki.flr.inputs import GetFLREventsInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600

def register_donki_flr_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register FLR MCP tools."""

    @mcp.tool()
    async def get_flr_events_tool(args: GetFLREventsInput) -> list[dict]:
        """Fetch NASA DONKI solar flare events for a date range.

        Returns flare records with activity ID, begin/peak/end times, class
        type (for example M or X class), source location, active region,
        linked events, instruments, and DONKI links. If dates are omitted,
        searches the last 30 days.

        Use for solar flare, radiation burst, active-region, heliophysics,
        aurora, or space-weather scene requests. Pair linked FLR events with
        CME tools when a flare appears to trigger a coronal mass ejection.
        """

        key_input = f"{args.start_date}|{args.end_date}"
        key = f"donki:get_flr_events:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_flr_events(config, args.start_date, args.end_date)
        ttl_seconds = SHORT_TTL if args.end_date is None or args.end_date == date.today() else LONG_TTL
        cache.set(key, response, ttl_seconds=ttl_seconds)
        
        return response
