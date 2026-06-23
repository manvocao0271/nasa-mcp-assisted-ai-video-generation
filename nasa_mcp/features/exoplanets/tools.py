"""MCP tool registration for NASA Exoplanet Archive."""

import hashlib

from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.features.exoplanets.api import get_exoplanet_stats, search_exoplanets
from nasa_mcp.features.exoplanets.inputs import SearchExoplanetsInput

SHORT_TTL = 24 * 3600   # 1 day — catalog is updated periodically
LONG_TTL = 7 * 24 * 3600  # 7 days — stats don't change often


def register_exoplanet_tools(mcp: FastMCP, config: Config, cache: Cache) -> None:
    """Register NASA Exoplanet Archive MCP tools."""

    @mcp.tool()
    async def search_exoplanets_tool(args: SearchExoplanetsInput) -> list[dict]:
        """Search NASA's confirmed exoplanet catalog for planets by name, host star, or discovery method.

        Returns up to *limit* exoplanet records (newest discoveries first),
        each containing:
        - ``pl_name``         — planet designation (e.g. "Kepler-452 b")
        - ``hostname``        — host star name
        - ``pl_orbper``       — orbital period in days
        - ``pl_rade``         — planet radius in Earth radii (null if unknown)
        - ``pl_masse``        — planet mass in Earth masses (null if unknown)
        - ``pl_eqt``          — equilibrium temperature in Kelvin (null if unknown)
        - ``disc_year``       — year of discovery
        - ``discoverymethod`` — how it was found ("Transit", "Radial Velocity", …)
        - ``disc_facility``   — telescope/facility name
        - ``st_spectype``     — host star spectral type (e.g. "G2V")
        - ``st_teff``         — host star effective temperature (K)

        Use for scripts or voiceovers about specific exoplanets or exoplanet demographics. Pair with search_image_library_tool to find artist's impression imagery of the system.

        Examples:
        - query="TRAPPIST"  → all TRAPPIST-1 planets
        - query="Kepler-452" → Kepler-452 b details
        - query="transit"   → transit-discovered planets
        - query="Proxima"   → planets around Proxima Centauri
        """
        key_input = f"{args.query}|{args.limit}"
        key = f"exo:search:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        results = await search_exoplanets(query=args.query, limit=args.limit)
        cache.set(key, results, ttl_seconds=SHORT_TTL)
        return results

    @mcp.tool()
    async def get_exoplanet_stats_tool() -> dict:
        """Return summary statistics for NASA's confirmed exoplanet catalog.

        Returns:
        - ``total_confirmed`` — total number of confirmed exoplanets
        - ``by_discovery_method`` — dict mapping each discovery method to its count
          (e.g. {"Transit": 3900, "Radial Velocity": 1100, …})

        Use to provide scale context in a script — e.g. "of the 5 000+ confirmed exoplanets, over 75 % were found by the transit method".
        """
        key = "exo:stats:global"
        cached = cache.get(key)
        if cached is not None:
            return cached

        result = await get_exoplanet_stats()
        cache.set(key, result, ttl_seconds=LONG_TTL)
        return result
