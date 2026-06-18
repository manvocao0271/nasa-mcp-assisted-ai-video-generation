"""MCP tool registration for GST."""

from datetime import date
import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.donki.gst.api import get_gst_events
from nasa_mcp.features.donki.gst.inputs import GetGSTEventsInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600


def register_donki_gst_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register GST MCP tools."""

    @mcp.tool()
    async def get_gst_events_tool(args: GetGSTEventsInput) -> list[dict]:
        """Fetch NASA DONKI geomagnetic storm events for a date range.

        Returns geomagnetic storm records with GST activity ID, start time, observed Kp index measurements, linked space-weather events, and DONKI links. If dates are omitted, searches the last 30 days.

        Use for geomagnetic storm, aurora, Earth impact, Kp index, magnetic disturbance, solar-storm aftermath, or space-weather scene requests. Pair linked GST events with CME tools when tracing a storm back to a coronal mass ejection.
        """

        key_input = f"{args.start_date}|{args.end_date}"
        key = f"donki:get_gst_events:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_gst_events(config, args.start_date, args.end_date)
        ttl_seconds = SHORT_TTL if args.end_date is None or args.end_date == date.today() else LONG_TTL
        cache.set(key, response, ttl_seconds=ttl_seconds)

        return response
