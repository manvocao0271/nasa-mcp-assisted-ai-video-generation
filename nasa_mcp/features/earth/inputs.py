"""Pydantic input models for EPIC (Earth Polychromatic Imaging Camera) MCP tools."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class GetEpicImagesInput(BaseModel):
    """Input validation for get_epic_images_tool."""

    target_date: date | None = Field(
        default=None,
        description=(
            "The date to retrieve EPIC full-disc Earth images for (YYYY-MM-DD). "
            "If omitted, returns the most recent available imagery."
        ),
    )
    collection: Literal["natural", "enhanced"] = Field(
        default="natural",
        description=(
            "Which image collection to query. "
            "'natural' returns true-colour RGB imagery; "
            "'enhanced' returns colour-enhanced imagery that highlights vegetation and other surface features."
        ),
    )


class GetEpicAvailableDatesInput(BaseModel):
    """Input validation for get_epic_available_dates_tool."""

    collection: Literal["natural", "enhanced"] = Field(
        default="natural",
        description=(
            "Which image collection to list available dates for. "
            "'natural' for true-colour imagery; 'enhanced' for colour-enhanced imagery."
        ),
    )
