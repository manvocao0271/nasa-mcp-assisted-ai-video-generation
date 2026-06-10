"""Pydantic input models for FLR MCP tools."""

from datetime import date

from pydantic import BaseModel, Field, model_validator


class GetFLREventsInput(BaseModel):
    """Input validation for get_flr_events_tool."""

    start_date: date | None = Field(
        default=None,
        description="Start date for the solar flare search, inclusive. Provide with end_date, or omit both to search the last 30 days.",
    )
    end_date: date | None = Field(
        default=None,
        description="End date for the solar flare search, inclusive. Provide with start_date, or omit both to search the last 30 days.",
    )

    @model_validator(mode="after")
    def check_range(self) -> "GetFLREventsInput":
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("Provide both start_date and end_date, or omit both.")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self
