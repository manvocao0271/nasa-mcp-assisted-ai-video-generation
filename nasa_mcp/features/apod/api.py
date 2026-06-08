"""Astronomy Picture of the Day (APOD) client.

Endpoints: https://api.nasa.gov/planetary/apod
"""

import asyncio
import httpx

from datetime import date
from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError


async def get_apod(config: Config, target_date: date | None = None) -> dict:
    """Fetch the Astronomy Picture of the Day for the given date (or today if omitted)."""

    params: dict[str, str] = {"api_key": config.nasa_api_key}
    if target_date is not None:
        params["date"] = target_date.isoformat()

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(
                "https://api.nasa.gov/planetary/apod",
                params=params,
            )
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(f"APOD returned {response.status_code}: {response.text}")


async def search_apod(config: Config, query: str, start_date: date, end_date: date) -> list[dict]:
    """Fetch APODs in a date range, filtered case-insensitively against title and explanation."""

    params: dict[str, str] = {
        "api_key": config.nasa_api_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        }

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(
                "https://api.nasa.gov/planetary/apod",
                params=params,
            )
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            q = query.lower()
            return [
                entry for entry in response.json()
                if q in entry.get("title", "").lower()
                or q in entry.get("explanation", "").lower()
            ]
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(f"APOD returned {response.status_code}: {response.text}")
        