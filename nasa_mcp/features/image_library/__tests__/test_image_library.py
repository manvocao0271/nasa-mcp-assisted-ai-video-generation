"""Unit tests for NASA Image and Video Library input models and API helpers."""

import pytest

from nasa_mcp.features.image_library.api import _build_large_url, _parse_items
from nasa_mcp.features.image_library.inputs import GetImageAssetInput, SearchImageLibraryInput


# ---------------------------------------------------------------------------
# Input model tests
# ---------------------------------------------------------------------------


def test_search_input_defaults() -> None:
    args = SearchImageLibraryInput(query="Mars rover")
    assert args.query == "Mars rover"
    assert args.page_size == 10
    assert args.year_start is None
    assert args.year_end is None


def test_search_input_accepts_year_range() -> None:
    args = SearchImageLibraryInput(query="Saturn", page_size=5, year_start=2000, year_end=2020)
    assert args.page_size == 5
    assert args.year_start == 2000
    assert args.year_end == 2020


def test_search_input_rejects_page_size_zero() -> None:
    with pytest.raises(Exception):
        SearchImageLibraryInput(query="test", page_size=0)


def test_search_input_rejects_page_size_above_100() -> None:
    with pytest.raises(Exception):
        SearchImageLibraryInput(query="test", page_size=101)


def test_get_asset_input_accepts_nasa_id() -> None:
    args = GetImageAssetInput(nasa_id="PIA23623")
    assert args.nasa_id == "PIA23623"


# ---------------------------------------------------------------------------
# API helper tests
# ---------------------------------------------------------------------------


def test_build_large_url_format() -> None:
    url = _build_large_url("PIA23623")
    assert url == "https://images-assets.nasa.gov/image/PIA23623/PIA23623~large.jpg"


def test_parse_items_returns_expected_keys() -> None:
    items = [
        {
            "href": "https://images-assets.nasa.gov/image/PIA123/collection.json",
            "data": [
                {
                    "nasa_id": "PIA123",
                    "title": "Mars Surface",
                    "description": "Red planet terrain",
                    "date_created": "2021-03-04T00:00:00Z",
                    "keywords": ["mars", "surface"],
                    "center": "JPL",
                    "media_type": "image",
                }
            ],
            "links": [
                {"href": "https://images-assets.nasa.gov/image/PIA123/PIA123~thumb.jpg", "rel": "preview"}
            ],
        }
    ]
    results = _parse_items(items)
    assert len(results) == 1
    r = results[0]
    assert r["nasa_id"] == "PIA123"
    assert r["title"] == "Mars Surface"
    assert r["url"] == "https://images-assets.nasa.gov/image/PIA123/PIA123~large.jpg"
    assert r["thumb_url"] == "https://images-assets.nasa.gov/image/PIA123/PIA123~thumb.jpg"
    assert r["keywords"] == ["mars", "surface"]


def test_parse_items_no_preview_link_falls_back_to_large_url() -> None:
    items = [
        {
            "href": "https://images-assets.nasa.gov/image/ABC/collection.json",
            "data": [{"nasa_id": "ABC", "title": "Test", "media_type": "image"}],
            "links": [],  # no preview link
        }
    ]
    results = _parse_items(items)
    assert results[0]["url"] == "https://images-assets.nasa.gov/image/ABC/ABC~large.jpg"
    assert results[0]["thumb_url"] == "https://images-assets.nasa.gov/image/ABC/ABC~large.jpg"


def test_parse_items_empty_returns_empty() -> None:
    assert _parse_items([]) == []
