"""Pydantic input models for NASA Exoplanet Archive MCP tools."""

from pydantic import BaseModel, Field


class SearchExoplanetsInput(BaseModel):
    """Input validation for search_exoplanets_tool."""

    query: str = Field(
        description=(
            "Search term matched against planet name, host star name, or discovery method. "
            "Examples: 'Kepler-452', 'TRAPPIST', 'transit', 'radial velocity', 'Proxima'."
        )
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of exoplanet records to return (1–200). Default 20.",
    )
