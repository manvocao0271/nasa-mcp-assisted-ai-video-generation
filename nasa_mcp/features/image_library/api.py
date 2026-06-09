"""NASA Image and Video Library API client.

Docs:   https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf
Search: https://images-api.nasa.gov/search
Assets: https://images-api.nasa.gov/asset/{nasa_id}

No API key required — the library is publicly accessible.
"""

import asyncio

import httpx

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError, NotFoundError, RateLimitError

_SEARCH_URL = "https://images-api.nasa.gov/search"
_ASSET_URL = "https://images-api.nasa.gov/asset"
_ASSETS_BASE = "https://images-assets.nasa.gov"


def _build_large_url(nasa_id: str) -> str:
    """Construct the large-JPEG URL for a NASA Image Library asset.

    NASA stores predictable size variants at:
      {base}/image/{nasa_id}/{nasa_id}~{size}.jpg
    'large' is always JPEG and suitable as a video first-frame reference.
    """
    return f"{_ASSETS_BASE}/image/{nasa_id}/{nasa_id}~large.jpg"


def _parse_items(items: list[dict]) -> list[dict]:
    """Extract a flat, tool-friendly dict from each collection item."""
    results = []
    for item in items:
        data = item.get("data", [{}])[0]
        links = item.get("links", [])

        nasa_id = data.get("nasa_id", "")
        thumb_url = next(
            (lnk["href"] for lnk in links if lnk.get("rel") == "preview"),
            None,
        )
        # Prefer large JPEG for usable ref images; fall back to preview thumb.
        url = _build_large_url(nasa_id) if nasa_id else (thumb_url or "")

        results.append(
            {
                "nasa_id": nasa_id,
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "date_created": data.get("date_created", ""),
                "keywords": data.get("keywords", []),
                "center": data.get("center", ""),
                "media_type": data.get("media_type", "image"),
                "url": url,
                "thumb_url": thumb_url or url,
            }
        )
    return results


async def search_image_library(
    config: Config,
    query: str,
    page_size: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
) -> list[dict]:
    """Search the NASA Image and Video Library for images matching *query*.

    Returns up to *page_size* image records, each with a usable ``url``
    pointing to a large JPEG suitable as a Wan first-frame reference.
    """
    params: dict[str, str | int] = {
        "q": query,
        "media_type": "image",
        "page_size": min(max(1, page_size), 100),
    }
    if year_start is not None:
        params["year_start"] = year_start
    if year_end is not None:
        params["year_end"] = year_end

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(_SEARCH_URL, params=params)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            items = response.json().get("collection", {}).get("items", [])
            return _parse_items(items)
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(f"No NASA images found for query '{query}'")
        case _:
            raise NasaApiError(
                f"NASA Image Library returned {response.status_code}: {response.text}"
            )


async def get_image_asset(config: Config, nasa_id: str) -> dict:
    """Retrieve all available size/format URLs for a single NASA image asset.

    Returns a dict with ``nasa_id``, ``large_url``, ``original_url``,
    ``medium_url``, ``thumb_url``, and ``all_urls`` (full list).
    """
    url = f"{_ASSET_URL}/{nasa_id}"

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(url)
        if response.status_code < 500:
            break
        if attempt < 2:
            await asyncio.sleep(2**attempt)

    match response.status_code:
        case status if 200 <= status < 300:
            items = response.json().get("collection", {}).get("items", [])
            all_urls = [item["href"] for item in items if "href" in item]
            pick = lambda suffix: next(  # noqa: E731
                (u for u in all_urls if u.endswith(suffix)), ""
            )
            return {
                "nasa_id": nasa_id,
                "large_url": pick("~large.jpg"),
                "original_url": pick("~orig.jpg"),
                "medium_url": pick("~medium.jpg"),
                "thumb_url": pick("~thumb.jpg"),
                "all_urls": all_urls,
                # Convenience: best usable image URL for video gen
                "url": pick("~large.jpg") or pick("~medium.jpg") or pick("~orig.jpg"),
            }
        case 429:
            raise RateLimitError(response.text)
        case 404:
            raise NotFoundError(f"No asset found for nasa_id '{nasa_id}'")
        case _:
            raise NasaApiError(
                f"NASA Image Library asset endpoint returned {response.status_code}: {response.text}"
            )
