"""MCP tool registration for APOD."""

from datetime import date, timedelta
import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.apod.api import get_apod, search_apod
from nasa_mcp.features.apod.inputs import GetApodInput, SearchApodInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600


def register_apod_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register APOD MCP tools."""

    @mcp.tool()
    async def get_apod_tool(args: GetApodInput) -> dict:
        """Fetch NASA's Astronomy Picture of the Day (APOD) for a specific date, or today if no date is given.

        Returns a dict with `title`, `explanation`, `url`, `hdurl`, `date`,
        `media_type` (usually `"image"`, sometimes `"video"`), and `copyright`
        (when applicable).

        Use this for questions like "show me today's astronomy picture", "what
        was the APOD on July 14 1998", or "find the explanation for yesterday's
        NASA image of the day". Valid dates range from 1995-06-16 to today.
        """

        key = f"apod:get_apod:{hashlib.sha256(str(args.target_date).encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_apod(config, args.target_date)
        ttl_seconds = (
            SHORT_TTL
            if args.target_date is None or args.target_date == date.today()
            else LONG_TTL
        )
        cache.set(key, response, ttl_seconds=ttl_seconds)
        return response

    @mcp.tool()
    async def search_apod_tool(args: SearchApodInput) -> list[dict]:
        """Search NASA's Astronomy Picture of the Day (APOD) archive for entries whose title or explanation matches a keyword, within a date range.

        Matches `query` case-insensitively against each entry's `title` and
        `explanation`. Returns a list of dicts, each containing `title`,
        `explanation`, `url`, `hdurl`, `date`, `media_type` (usually `"image"`,
        sometimes `"video"`), and `copyright` (when applicable). The list is
        empty if nothing matches.

        If `start_date` and `end_date` are omitted, defaults to the last 30
        days. Valid dates range from 1995-06-16 to today.

        Use this for questions like "find APODs about black holes from 2024",
        "show me Mars photos from June 2023", or "any APODs mentioning Hubble
        in the last month?".
        """

        start_date = args.start_date or (date.today() - timedelta(days=30))
        end_date = args.end_date or date.today()

        key_input = f"{args.query}|{start_date}|{end_date}"
        key = f"apod:search_apod:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await search_apod(config, args.query, start_date, end_date)
        ttl_seconds = SHORT_TTL if end_date == date.today() else LONG_TTL
        cache.set(key, response, ttl_seconds=ttl_seconds)
        return response
