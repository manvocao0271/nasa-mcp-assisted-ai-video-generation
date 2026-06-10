"""Pydantic input models for CME MCP tools."""

from datetime import date

from pydantic import BaseModel, Field, model_validator


class GetCMEEventsInput(BaseModel):
    """Input validation for get_cme_events_tool."""

    start_date: date | None = Field(
        default=None,
        description="The date to start for a ranged search, inclusive. If omitted, defaults to 30 days before today.",
    )
    end_date: date | None = Field(
        default=None,
        description="The date to end for a ranged search, inclusive. If omitted, defaults to today.",
    )

    @model_validator(mode="after")
    def check_range(self) -> "GetCMEEventsInput":
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("Provide both start_date and end_date, or omit both.")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self