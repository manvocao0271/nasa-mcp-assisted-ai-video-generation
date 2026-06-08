"""Pydantic input models for APOD MCP tools."""

from datetime import date

from pydantic import BaseModel, Field


class GetApodInput(BaseModel):
    """Input validation for get_apod_tool."""

    target_date: date | None = Field(
        default=None,
        description="The date to look up ranging 1995-06-16 to present day. If omitted, return today's APOD.",
    )


class SearchApodInput(BaseModel):
    """Input validation for search_apod_tool."""

    query: str = Field(
        description="The search input to match case-insensitively with title and/or explanation."
    )
    start_date: date | None = Field(
        default=None,
        description="The date to start for a ranged search, inclusive. If omitted, defaults to 30 days before today.",
    )
    end_date: date | None = Field(
        default=None,
        description="The date to end for a ranged search, inclusive. If omitted, defaults to today.",
    )
