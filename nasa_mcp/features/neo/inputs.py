"""Pydantic input models for NeoWs MCP tools."""

from datetime import date, timedelta

from pydantic import BaseModel, Field, model_validator


class GetNeoFeedInput(BaseModel):
    """Input validation for get_neo_feed_tool."""

    start_date: date = Field(
        description="The beginning date to start for a ranged search, inclusive."
    )
    end_date: date | None = Field(
        default=None,
        description="The final date to start for a ranged search, inclusive. If omitted, the final search date range is seven days after the starting date."
    )

    @model_validator(mode="after")
    def check_range(self) -> "GetNeoFeedInput":
        end = self.end_date or (self.start_date + timedelta(days=7))
        if (end - self.start_date).days > 7:
            raise ValueError("Date range cannot exceed 7 days (NASA API limit).")
        return self


class GetNeoLookupInput(BaseModel):
    """Input validation for get_neo_lookup_tool."""

    asteroid_id: str = Field(
        description="The SPK-ID of the asteroid to look up. For example '3542519'."
    )