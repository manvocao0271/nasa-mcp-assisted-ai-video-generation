"""MCP tool registration for EPIC (Earth Polychromatic Imaging Camera)."""

import hashlib
from datetime import date

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.earth.api import get_epic_available_dates, get_epic_images
from nasa_mcp.features.earth.inputs import GetEpicAvailableDatesInput, GetEpicImagesInput

LONG_TTL = 365 * 24 * 3600
SHORT_TTL = 4 * 3600


def register_earth_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register EPIC Earth imagery MCP tools."""

    @mcp.tool()
    async def get_epic_images_tool(args: GetEpicImagesInput) -> list[dict]:
        """Fetch full-disc Earth images captured by DSCOVR's EPIC camera for a specific date, or the most recent available date if none is given.

        Returns a list of image records. Each record contains:
        - ``identifier``: unique image ID
        - ``caption``: short description of the image
        - ``image``: raw filename (used to construct the URL)
        - ``date``: capture timestamp (``"YYYY-MM-DD HH:MM:SS"``)
        - ``centroid_coordinates``: ``{"lat": float, "lon": float}`` — the point on Earth the satellite is centred on
        - ``jpg_url``: direct link to the half-resolution JPEG on the EPIC archive
        - ``dscovr_j2000_position``, ``lunar_j2000_position``, ``sun_j2000_position``: spacecraft and body positions in J2000 coordinates
        - ``attitude_quaternions``: satellite orientation

        Use ``collection="natural"`` (default) for true-colour imagery; ``"enhanced"`` for colour-enhanced imagery that highlights vegetation.

        Use this for questions like "show me what Earth looked like today from space", "get a full-disc Earth photo from 2024-01-01", or "what does the EPIC camera see?". EPIC does not provide close-up views of specific locations — every image is the entire Earth disc from ~1.5 million km away.
        """
        
        key = f"earth:get_epic_images:{hashlib.sha256(f'{args.target_date}|{args.collection}'.encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        records = await get_epic_images(config, args.collection, args.target_date)
        ttl = (
            SHORT_TTL
            if args.target_date is None or args.target_date == date.today()
            else LONG_TTL
        )
        cache.set(key, records, ttl_seconds=ttl)
        return records

    @mcp.tool()
    async def get_epic_available_dates_tool(args: GetEpicAvailableDatesInput) -> list[str]:
        """List all dates that have EPIC full-disc Earth imagery available for a given collection.

        Returns a list of ISO date strings (``"YYYY-MM-DD"``) in ascending order, starting from 2015-06-13. There are typically multiple captures per day; this tool returns the dates, not the individual images — call ``get_epic_images_tool`` with a specific date to retrieve the images.

        Use this to discover what dates are available before calling ``get_epic_images_tool``, or to answer questions like "when was the first EPIC image taken?", "how many days of Earth imagery exist?", or "is there EPIC imagery from 2017-08-21 (the solar eclipse)?".
        """

        key = f"earth:get_epic_available_dates:{args.collection}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        dates = await get_epic_available_dates(config, args.collection)
        cache.set(key, dates, ttl_seconds=SHORT_TTL)
        return dates
