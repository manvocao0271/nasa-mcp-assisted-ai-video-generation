"""Unit tests for EPIC Earth input models and URL builder."""

from datetime import date

import pytest

from nasa_mcp.features.earth.api import build_image_url
from nasa_mcp.features.earth.inputs import GetEpicAvailableDatesInput, GetEpicImagesInput


def test_get_epic_images_input_defaults_to_natural_and_no_date() -> None:
    args = GetEpicImagesInput()

    assert args.collection == "natural"
    assert args.target_date is None


def test_get_epic_images_input_accepts_date_and_collection() -> None:
    args = GetEpicImagesInput(target_date=date(2024, 1, 1), collection="enhanced")

    assert args.target_date == date(2024, 1, 1)
    assert args.collection == "enhanced"


def test_get_epic_images_input_rejects_invalid_collection() -> None:
    with pytest.raises(Exception):
        GetEpicImagesInput(collection="aerosol")  # type: ignore[arg-type]


def test_get_epic_available_dates_input_defaults_to_natural() -> None:
    args = GetEpicAvailableDatesInput()

    assert args.collection == "natural"


def test_build_image_url_constructs_jpg_url_correctly() -> None:
    url = build_image_url(
        image_name="epic_1b_20161031074844",
        collection="natural",
        image_date="2016-10-31",
    )

    assert url == (
        "https://epic.gsfc.nasa.gov/archive/natural/2016/10/31/jpg/epic_1b_20161031074844.jpg"
    )


def test_build_image_url_constructs_png_url_correctly() -> None:
    url = build_image_url(
        image_name="epic_1b_20161031074844",
        collection="natural",
        image_date="2016-10-31",
        format="png",
    )

    assert url == (
        "https://epic.gsfc.nasa.gov/archive/natural/2016/10/31/png/epic_1b_20161031074844.png"
    )


def test_build_image_url_handles_enhanced_collection() -> None:
    url = build_image_url(
        image_name="epic_RGB_20161031074844",
        collection="enhanced",
        image_date="2016-10-31",
    )

    assert url == (
        "https://epic.gsfc.nasa.gov/archive/enhanced/2016/10/31/jpg/epic_RGB_20161031074844.jpg"
    )
