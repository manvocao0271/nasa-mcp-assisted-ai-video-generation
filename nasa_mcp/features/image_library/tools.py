"""MCP tool registration for NASA Image and Video Library."""

import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.image_library.api import get_image_asset, search_image_library
from nasa_mcp.features.image_library.inputs import GetImageAssetInput, SearchImageLibraryInput

SHORT_TTL = 6 * 3600   # 6 h — search results can shift as new images are added
LONG_TTL = 7 * 24 * 3600  # 7 days — individual assets are immutable


def register_image_library_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register NASA Image and Video Library MCP tools."""

    @mcp.tool()
    async def search_image_library_tool(args: SearchImageLibraryInput) -> list[dict]:
        """Search NASA's Image and Video Library (140 000+ assets) for images matching keywords.

        Returns a list of image records, each containing:
        - ``nasa_id``   — unique asset identifier (use with get_image_asset_tool for full sizes)
        - ``title``     — human-readable caption
        - ``description`` — extended caption / press release excerpt
        - ``date_created`` — ISO-8601 timestamp
        - ``keywords``  — list of subject tags
        - ``center``    — NASA centre that produced the image (JPL, GSFC, JSC, …)
        - ``url``       — large JPEG suitable as a Wan video first-frame reference
        - ``thumb_url`` — small preview thumbnail

        Use for any topic that needs high-quality NASA photography: Mars rover
        surface shots, Saturn flyby imagery, Earth from ISS, deep-field Hubble
        images, rocket launches, spacewalks, etc.

        Examples:
        - query="Mars surface Perseverance rover" → recent Mars landscape photos
        - query="Saturn rings Cassini" → Cassini mission Saturn imagery
        - query="ISS Earth orbit" → station-to-Earth vantage shots
        - query="nebula Hubble infrared" → Hubble deep-field nebulae
        """
        key_input = f"{args.query}|{args.page_size}|{args.year_start}|{args.year_end}"
        key = f"imglib:search:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        results = await search_image_library(
            config,
            query=args.query,
            page_size=args.page_size,
            year_start=args.year_start,
            year_end=args.year_end,
        )
        cache.set(key, results, ttl_seconds=SHORT_TTL)
        return results

    @mcp.tool()
    async def get_image_asset_tool(args: GetImageAssetInput) -> dict:
        """Retrieve all available size variants for a specific NASA image asset by its ID.

        Given a ``nasa_id`` (returned by search_image_library_tool), returns
        every downloadable URL for that image:
        - ``url``          — best usable URL (large JPEG when available)
        - ``large_url``    — ~1800 px JPEG
        - ``medium_url``   — ~800 px JPEG
        - ``thumb_url``    — small thumbnail
        - ``original_url`` — full-resolution original (may be very large / non-JPEG)
        - ``all_urls``     — complete list of all hosted file URLs

        Use this when you need the highest-quality version of an image found via
        search_image_library_tool, or when you want to confirm the exact URL
        before passing it to a video generation tool.
        """
        key = f"imglib:asset:{hashlib.sha256(args.nasa_id.encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        result = await get_image_asset(config, args.nasa_id)
        cache.set(key, result, ttl_seconds=LONG_TTL)
        return result
