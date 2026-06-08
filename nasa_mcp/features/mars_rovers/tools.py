"""MCP tool registration for Mars rovers."""

import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.mars_rovers.api import get_rover_manifest, get_rover_photos
from nasa_mcp.features.mars_rovers.inputs import (
    GetRoverManifestInput,
    GetRoverPhotosInput,
)

LONG_TTL = 365 * 24 * 3600


def register_mars_rover_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register Mars rover MCP tools."""

    @mcp.tool()
    async def get_rover_photos_tool(args: GetRoverPhotosInput) -> dict:
        """Fetch photos taken by a NASA Mars rover on a specific sol (Martian day) or Earth date, optionally filtered by camera.

        Returns a dict with a `photos` list. Each entry contains `id`, `sol`,
        `earth_date`, `img_src` (CDN URL to the JPEG), `camera` (with `name`,
        `full_name`, abbreviation), and `rover` (with `name`, `landing_date`,
        `launch_date`, `status`). Empty list if no photos match.

        Provide exactly one of `sol` or `earth_date`. Supported rovers:
        curiosity, perseverance, spirit, opportunity. Optional `camera` codes
        include FHAZ (front hazard), RHAZ (rear hazard), MAST, NAVCAM, CHEMCAM,
        MAHLI, MARDI, PANCAM, MINITES — not every rover carries every camera.

        Use this for questions like "show me what Curiosity saw on sol 1000",
        "what did Perseverance photograph on 2024-06-15", or "Spirit's MAST
        camera images from sol 500".
        """

        if args.sol is not None:
            date_key = f"sol:{args.sol}"
        else:
            assert args.earth_date is not None
            date_key = f"earth_date:{args.earth_date.isoformat()}"

        camera_key = args.camera or "all"
        key_input = f"{args.rover}|{date_key}|{camera_key}"
        key = f"mars_rover_photos:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_rover_photos(
            config,
            rover_name=args.rover,
            sol=args.sol,
            earth_date=args.earth_date,
            camera=args.camera,
        )
        cache.set(key, response, ttl_seconds=LONG_TTL)
        return response

    @mcp.tool()
    async def get_rover_manifest_tool(args: GetRoverManifestInput) -> dict:
        """Fetch the mission manifest for a NASA Mars rover — landing/launch dates, status, total photo count, and per-sol summaries.

        Returns a dict with a `photo_manifest` object containing `name`,
        `landing_date`, `launch_date`, `status` (`"active"` or `"complete"`),
        `max_sol` (highest sol with photos), `max_date` (latest Earth date with
        photos), `total_photos`, and `photos` — a list summarizing each sol
        with `sol`, `earth_date`, `total_photos`, and `cameras` (camera codes
        used that sol).

        Supported rovers: curiosity, perseverance, spirit, opportunity.

        Use this for questions like "how many photos has Curiosity taken in
        total", "when did Perseverance land", "what's the latest sol Spirit
        has photos for", or "which cameras did Opportunity have".
        """

        key = f"mars_rover_manifest:{args.rover}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        response = await get_rover_manifest(config, args.rover)
        cache.set(key, response, ttl_seconds=LONG_TTL)
        return response
