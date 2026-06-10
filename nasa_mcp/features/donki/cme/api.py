"""Coronal Mass Ejection (CME) client.

Endpoints: https://api.nasa.gov/DONKI/CME
"""

import asyncio
from datetime import date, timedelta

import httpx

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError


async def get_cme_events(config: Config, start_date: date | None = None, end_date: date | None = None) -> list[dict]:
    """Fetch Coronal Mass Ejection events for a date range, or the last month if omitted."""

    params: dict[str, str] = {
        "api_key": config.nasa_api_key,
    }

    if start_date is None and end_date is None:
        params["startDate"] = (date.today() - timedelta(days=30)).isoformat()
        params["endDate"] = date.today().isoformat()
    elif start_date is None or end_date is None:
        raise NasaApiError("DONKI-CME requires both start_date and end_date, or neither.")
    elif end_date < start_date:
        raise NasaApiError("DONKI-CME end_date must be on or after start_date.")
    else:
        params["startDate"] = start_date.isoformat()
        params["endDate"] = end_date.isoformat()
    
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(
                "https://api.nasa.gov/DONKI/CME",
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
            raise NasaApiError(f"DONKI-CME returned {response.status_code}: {response.text}")
        

