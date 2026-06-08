"""Unit tests for Mars rover input models."""

from datetime import date

import pytest
from pydantic import ValidationError

from nasa_mcp.features.mars_rovers.inputs import (
    GetRoverManifestInput,
    GetRoverPhotosInput,
)


def test_rover_photos_accepts_sol_selector() -> None:
    args = GetRoverPhotosInput(rover="Curiosity", sol=1000, camera="NAVCAM")

    assert args.rover == "curiosity"
    assert args.sol == 1000
    assert args.earth_date is None
    assert args.camera == "navcam"


def test_rover_photos_accepts_earth_date_selector() -> None:
    args = GetRoverPhotosInput(rover="perseverance", earth_date=date(2024, 1, 1))

    assert args.rover == "perseverance"
    assert args.sol is None
    assert args.earth_date == date(2024, 1, 1)


def test_rover_photos_rejects_missing_date_selector() -> None:
    with pytest.raises(ValidationError, match="Provide exactly one of sol or earth_date"):
        GetRoverPhotosInput(rover="curiosity")


def test_rover_photos_rejects_multiple_date_selectors() -> None:
    with pytest.raises(ValidationError, match="Provide exactly one of sol or earth_date"):
        GetRoverPhotosInput(
            rover="curiosity",
            sol=1000,
            earth_date=date(2024, 1, 1),
        )


def test_rover_photos_rejects_negative_sol() -> None:
    with pytest.raises(ValidationError):
        GetRoverPhotosInput(rover="curiosity", sol=-1)


def test_rover_photos_rejects_unsupported_rover() -> None:
    with pytest.raises(ValidationError):
        GetRoverPhotosInput(rover="sojourner", sol=1)


def test_rover_manifest_accepts_supported_rover() -> None:
    args = GetRoverManifestInput(rover="Opportunity")

    assert args.rover == "opportunity"


def test_rover_manifest_rejects_unsupported_rover() -> None:
    with pytest.raises(ValidationError):
        GetRoverManifestInput(rover="sojourner")
