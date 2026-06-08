"""Mars rovers feature package."""

from nasa_mcp.features.mars_rovers.inputs import (
    GetRoverManifestInput,
    GetRoverPhotosInput,
    SupportedRoverName,
)
from nasa_mcp.features.mars_rovers.tools import register_mars_rover_tools

__all__ = [
    "GetRoverManifestInput",
    "GetRoverPhotosInput",
    "SupportedRoverName",
    "register_mars_rover_tools",
]
