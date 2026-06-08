"""Live-network tests that hit real NASA APIs. Run with `pytest -m live`.

These confirm that the client speaks the protocol NASA actually serves. They
do not gate on NASA being up — they're skipped by default. To run them
explicitly: `uv run pytest -m live`.
"""

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest
from dotenv import load_dotenv

from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError
from nasa_mcp.features.apod.api import get_apod, search_apod
from nasa_mcp.features.earth.api import get_epic_available_dates, get_epic_images

pytestmark = pytest.mark.asyncio


def _live_config() -> Config:
    """Build a Config with the developer's real NASA_API_KEY for live tests."""
    
    load_dotenv()
    return Config(
        nasa_api_key=os.environ.get("NASA_API_KEY", "DEMO_KEY"),
        cache_path=Path(tempfile.gettempdir()) / "nasa-live-tests.sqlite3",
        request_timeout=60.0,
    )


@pytest.mark.live
async def test_get_apod_returns_expected_fields() -> None:
    try:
        result = await get_apod(_live_config(), date(2024, 1, 15))
    except NasaApiError as e:
        pytest.skip(f"NASA APOD unavailable upstream: {e}")

    assert isinstance(result, dict)
    assert "title" in result
    assert "url" in result
    assert result.get("date") == "2024-01-15"


@pytest.mark.live
async def test_search_apod_filters_by_keyword() -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    
    try:
        results = await search_apod(_live_config(), query="galaxy", start_date=start, end_date=end)
    except NasaApiError as e:
        pytest.skip(f"NASA APOD unavailable upstream: {e}")

    assert isinstance(results, list)
    for entry in results:
        body = (entry.get("title", "") + entry.get("explanation", "")).lower()
        assert "galaxy" in body


@pytest.mark.live
async def test_get_epic_images_returns_expected_fields() -> None:
    try:
        records = await get_epic_images(_live_config(), "natural")
    except NasaApiError as e:
        pytest.skip(f"EPIC API unavailable upstream: {e}")

    assert isinstance(records, list)
    assert len(records) > 0
    first = records[0]
    assert "identifier" in first
    assert "caption" in first
    assert "date" in first
    assert "centroid_coordinates" in first
    assert "jpg_url" in first
    assert first["jpg_url"].startswith("https://epic.gsfc.nasa.gov/archive/natural/")
    assert first["jpg_url"].endswith(".jpg")


@pytest.mark.live
async def test_get_epic_available_dates_returns_sorted_list() -> None:
    try:
        dates = await get_epic_available_dates(_live_config(), "natural")
    except NasaApiError as e:
        pytest.skip(f"EPIC API unavailable upstream: {e}")

    assert isinstance(dates, list)
    assert len(dates) > 0
    # Archive starts 2015-06-13
    assert dates[0] >= "2015-06-13"
    # List should be in ascending order
    assert dates == sorted(dates)
