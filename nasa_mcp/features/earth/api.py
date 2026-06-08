"""EPIC (Earth Polychromatic Imaging Camera) API client.

Endpoints: https://api.nasa.gov/EPIC/api/
Image archive: https://epic.gsfc.nasa.gov/archive/
API docs: https://epic.gsfc.nasa.gov/about/api
"""

import asyncio
from datetime import date
from typing import Literal

import httpx

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError

_API_BASE = "https://api.nasa.gov/EPIC/api"
_ARCHIVE_BASE = "https://epic.gsfc.nasa.gov/archive"


def build_image_url(
    image_name: str,
    collection: str,
    image_date: str,
    format: Literal["jpg", "png", "thumbs"] = "jpg",
) -> str:
    """Construct the archive URL for an EPIC image on epic.gsfc.nasa.gov."""

    year, month, day = image_date[:10].split("-")
    return f"{_ARCHIVE_BASE}/{collection}/{year}/{month}/{day}/{format}/{image_name}.{format}"


async def get_epic_images(
    config: Config,
    collection: str,
    target_date: date | None = None,
) -> list[dict]:
    """Fetch EPIC full-disc Earth image metadata for a date (or the latest available date)."""

    if target_date is not None:
        url = f"{_API_BASE}/{collection}/date/{target_date.isoformat()}"
    else:
        url = f"{_API_BASE}/{collection}"

    params = {"api_key": config.nasa_api_key}

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(url, params=params)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            records = response.json()
            for record in records:
                record["jpg_url"] = build_image_url(
                    record["image"], collection, record["date"]
                )
            return records
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(
                f"No EPIC {collection} imagery available"
                + (f" for {target_date}" if target_date else "")
                + f": {response.text}"
            )
        case _:
            raise NasaApiError(
                f"EPIC API returned {response.status_code}: {response.text}"
            )


async def get_epic_available_dates(config: Config, collection: str) -> list[str]:
    """Fetch all dates that have EPIC imagery for the given collection."""

    url = f"{_API_BASE}/{collection}/available"
    params = {"api_key": config.nasa_api_key}

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(url, params=params)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case _:
            raise NasaApiError(
                f"EPIC API returned {response.status_code}: {response.text}"
            )
