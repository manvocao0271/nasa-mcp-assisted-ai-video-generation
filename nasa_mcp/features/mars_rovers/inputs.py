"""Pydantic input models for Mars rover MCP tools."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SupportedRoverName = Literal["curiosity", "perseverance", "spirit", "opportunity"]


def _normalize_optional_string(value: str | None) -> str | None:
    """Normalize optional string inputs used by NASA rover endpoints."""
    if value is None:
        return None

    normalized = value.strip().lower()
    return normalized or None


class GetRoverPhotosInput(BaseModel):
    """Input validation for get_rover_photos_tool."""

    rover: SupportedRoverName = Field(
        description="Mars rover name to fetch photos from. Supported values: curiosity, perseverance, spirit, opportunity.",
    )
    sol: int | None = Field(
        default=None,
        ge=0,
        description="Martian sol to fetch photos for. Provide exactly one of sol or earth_date.",
    )
    earth_date: date | None = Field(
        default=None,
        description="Earth calendar date to fetch photos for. Provide exactly one of sol or earth_date.",
    )
    camera: str | None = Field(
        default=None,
        description="Optional rover camera abbreviation filter, such as FHAZ, RHAZ, NAVCAM, MAST, CHEMCAM, MAHLI, MARDI, PANCAM, or MINITES.",
    )

    @field_validator("rover", mode="before")
    @classmethod
    def normalize_rover(cls, value: str) -> str:
        """Allow case-insensitive rover names while exposing strict enum values."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("camera", mode="before")
    @classmethod
    def normalize_camera(cls, value: str | None) -> str | None:
        """Normalize optional camera abbreviations before the API request."""
        return _normalize_optional_string(value)

    @model_validator(mode="after")
    def validate_date_selection(self) -> "GetRoverPhotosInput":
        """Require one, and only one, Mars date selector."""
        if (self.sol is None) == (self.earth_date is None):
            raise ValueError("Provide exactly one of sol or earth_date.")
        return self


class GetRoverManifestInput(BaseModel):
    """Input validation for get_rover_manifest_tool."""

    rover: SupportedRoverName = Field(
        description="Mars rover name to fetch the mission manifest for. Supported values: curiosity, perseverance, spirit, opportunity.",
    )

    @field_validator("rover", mode="before")
    @classmethod
    def normalize_rover(cls, value: str) -> str:
        """Allow case-insensitive rover names while exposing strict enum values."""
        return value.strip().lower() if isinstance(value, str) else value
