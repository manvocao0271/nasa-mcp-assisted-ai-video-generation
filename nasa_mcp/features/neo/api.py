"""Near Earth Object Web Service (NeoWs) client.

Endpoints:
- https://api.nasa.gov/neo/rest/v1/feed/
- https://api.nasa.gov/neo/rest/v1/neo/{asteroid_id}
"""

import asyncio
from datetime import date, timedelta

import httpx

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError


async def get_neo_feed(config: Config, start_date: date, end_date: date | None = None) -> dict:
    """Fetch all asteroids approaching in a date range."""

    params: dict[str, str] = {
        "api_key": config.nasa_api_key,
        "start_date": start_date.isoformat(),
    }

    if end_date is not None:
        params["end_date"] = end_date.isoformat()
    else:
        params["end_date"] = (start_date + timedelta(days=7)).isoformat()
    
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(
                "https://api.nasa.gov/neo/rest/v1/feed",
                params=params,
            )
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(f"NEO returned {response.status_code}: {response.text}")
        

async def get_neo_lookup(config: Config, asteroid_id: str) -> dict:
    """Fetch full data for one asteroid."""

    params: dict[str, str] = {
        "api_key": config.nasa_api_key,
    }

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(
                f"https://api.nasa.gov/neo/rest/v1/neo/{asteroid_id}",
                params=params,
            )
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(f"NEO returned {response.status_code}: {response.text}")
    