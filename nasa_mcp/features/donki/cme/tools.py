"""MCP tool registration for CME."""

from datetime import date
import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.donki.cme.api import get_cme_events
from nasa_mcp.features.donki.cme.inputs import GetCMEEventsInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600

def register_donki_cme_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register CME MCP tools."""

    @mcp.tool()
    async def get_cme_events_tool(args: GetCMEEventsInput) -> list[dict]:
        """Fetch NASA DONKI Coronal Mass Ejection events for a date range.

        Returns CME records with activity ID, start time, source location,
        active region, instruments, `cmeAnalyses`, ENLIL model runs, linked
        events, and DONKI links. If dates are omitted, searches the last 30
        days.

        Use for solar storm, coronal mass ejection, heliophysics, aurora, or space-weather scene requests.
        """

        key_input = f"{args.start_date}|{args.end_date}"
        key = f"donki:get_cme_events:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_cme_events(config, args.start_date, args.end_date)
        ttl_seconds = SHORT_TTL if args.end_date is None or args.end_date == date.today() else LONG_TTL
        cache.set(key, response, ttl_seconds=ttl_seconds)
        
        return response
