"""NASA Exoplanet Archive API client.

Endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
Docs:     https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

No API key required.  Uses the IPAC TAP (Table Access Protocol) endpoint
with ADQL queries, returning JSON.

The Planetary Systems (ps) table is the canonical confirmed-exoplanet table.
"""

import asyncio
import urllib.parse

import httpx

from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError

_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns returned for each planet — informative but not overwhelming.
_COLUMNS = (
    "pl_name, hostname, sy_snum, sy_pnum, "
    "pl_orbper, pl_rade, pl_masse, pl_eqt, "
    "disc_year, discoverymethod, disc_facility, "
    "st_spectype, st_teff, st_rad, st_mass"
)


async def search_exoplanets(
    query: str,
    limit: int = 20,
) -> list[dict]:
    """Search confirmed exoplanets by name, host star, or discovery method.

    Uses a case-insensitive LIKE match against planet name and host star name. Returns up to *limit* records sorted by discovery year (newest first).
    """

    # Escape single quotes to prevent ADQL injection
    safe_q = query.replace("'", "''")
    adql = (
        f"SELECT TOP {limit} {_COLUMNS} "
        f"FROM ps "
        f"WHERE default_flag=1 "
        f"AND (UPPER(pl_name) LIKE UPPER('%{safe_q}%') "
        f"  OR UPPER(hostname) LIKE UPPER('%{safe_q}%') "
        f"  OR UPPER(discoverymethod) LIKE UPPER('%{safe_q}%')) "
        f"ORDER BY disc_year DESC"
    )
    params = {"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"}

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(_TAP_URL, params=params)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            data = response.json()
            # TAP JSON format: {"fields": [...], "data": [[...], ...]}
            # or may return a list of dicts directly depending on endpoint version
            if isinstance(data, list):
                return data
            fields = [f["name"] for f in data.get("fields", [])]
            rows = data.get("data", [])
            if fields and rows:
                return [dict(zip(fields, row)) for row in rows]
            return []
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(f"No exoplanets found for query '{query}'")
        case _:
            raise NasaApiError(
                f"Exoplanet Archive returned {response.status_code}: {response.text}"
            )


async def get_exoplanet_stats() -> dict:
    """Return aggregate statistics from the confirmed exoplanet catalog.

    Counts by discovery method and total confirmed planet count — useful for context in scripts about exoplanets.
    """

    adql = (
        "SELECT discoverymethod, COUNT(*) as count "
        "FROM ps "
        "WHERE default_flag=1 "
        "GROUP BY discoverymethod "
        "ORDER BY count DESC"
    )
    params = {"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"}

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(_TAP_URL, params=params)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            data = response.json()
            if isinstance(data, list):
                by_method = {row.get("discoverymethod", "?"): row.get("count", 0) for row in data}
            else:
                fields = [f["name"] for f in data.get("fields", [])]
                rows = data.get("data", [])
                by_method = {row[0]: row[1] for row in rows} if rows else {}
            return {
                "total_confirmed": sum(by_method.values()),
                "by_discovery_method": by_method,
            }
        case _:
            raise NasaApiError(
                f"Exoplanet Archive stats returned {response.status_code}: {response.text}"
            )
