"""Mars rover photos client (Curiosity, Perseverance, Spirit, Opportunity).

Endpoints: https://api.nasa.gov/mars-photos/api/v1/
"""

from datetime import date

import httpx

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError


def _check_availability(response: httpx.Response) -> None:
    """Raise NasaApiError if the response is HTML - NASA's Mars Photos backend is archived."""

    if response.headers.get("content-type", "").startswith("text/html"):
        raise NasaApiError(
            "NASA's Mars Photos API is currently unavailable. The upstream backend (corincerami/mars-photo-api on Heroku) has been archived. Track status at https://github.com/corincerami/mars-photo-api."
        )


async def get_rover_photos(
    config: Config,
    rover_name: str,
    sol: int | None = None,
    earth_date: date | None = None,
    camera: str | None = None,
) -> dict:
    """Fetch photos taken by a given Mars rover on a specific sol or Earth date."""

    params: dict[str, str] = {"api_key": config.nasa_api_key}
    if sol is not None:
        params["sol"] = str(sol)
    if earth_date is not None:
        params["earth_date"] = earth_date.isoformat()
    if camera is not None:
        params["camera"] = camera

    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        response = await client.get(
            f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover_name}/photos",
            params=params,
        )
    
    _check_availability(response)

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(
                f"Rover photos returned {response.status_code}: {response.text}"
            )


async def get_rover_manifest(config: Config, rover_name: str) -> dict:
    """Fetch the manifest for a given Mars rover."""

    params: dict[str, str] = {"api_key": config.nasa_api_key}

    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        response = await client.get(
            f"https://api.nasa.gov/mars-photos/api/v1/manifests/{rover_name}",
            params=params,
        )

    _check_availability(response)

    match response.status_code:
        case status if 200 <= status < 300:
            return response.json()
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(response.text)
        case _:
            raise NasaApiError(
                f"Rover manifest returned {response.status_code}: {response.text}"
            )
