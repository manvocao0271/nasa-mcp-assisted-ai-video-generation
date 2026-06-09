"""Pydantic input models for NASA Image and Video Library MCP tools."""

from pydantic import BaseModel, Field


class SearchImageLibraryInput(BaseModel):
    """Input validation for search_image_library_tool."""

    query: str = Field(
        description=(
            "Search keywords. Be specific: e.g. 'Mars surface rover', "
            "'Saturn rings Cassini', 'Hubble deep field galaxy'."
        )
    )
    page_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of image results to return (1–100). Default 10.",
    )
    year_start: int | None = Field(
        default=None,
        description="Restrict results to images created on or after this year (e.g. 2000).",
    )
    year_end: int | None = Field(
        default=None,
        description="Restrict results to images created on or before this year (e.g. 2024).",
    )


class GetImageAssetInput(BaseModel):
    """Input validation for get_image_asset_tool."""

    nasa_id: str = Field(
        description=(
            "The NASA asset ID returned by search_image_library_tool "
            "(e.g. 'PIA23623', 'GSFC_20171208_Archive_e000200'). "
            "Returns all size variants including the original."
        )
    )
