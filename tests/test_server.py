"""Integration tests: tool registration and LLM-facing description quality."""

import pytest
from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.asyncio

EXPECTED_TOOLS = {
    "get_apod_tool",
    "search_apod_tool",
    "get_epic_images_tool",
    "get_epic_available_dates_tool",
    "get_rover_photos_tool",
    "get_rover_manifest_tool",
    "get_neo_feed_tool",
    "get_neo_lookup_tool",
}

MIN_DESCRIPTION_LENGTH = 150


async def test_all_expected_tools_are_registered(mcp_with_tools: FastMCP) -> None:
    """Each feature module registers its tools on the shared FastMCP instance."""
    tools = await mcp_with_tools.list_tools()
    names = {tool.name for tool in tools}

    missing = EXPECTED_TOOLS - names
    assert not missing, f"Missing registered tools: {missing}"


async def test_every_tool_has_a_substantive_description(mcp_with_tools: FastMCP) -> None:
    """Tool descriptions are surfaced to the LLM and must be specific, not stubs."""
    tools = await mcp_with_tools.list_tools()

    short_descriptions = [
        (tool.name, len(tool.description or ""))
        for tool in tools
        if not tool.description or len(tool.description) < MIN_DESCRIPTION_LENGTH
    ]
    assert not short_descriptions, (
        f"Tools missing rich LLM-facing descriptions (need ≥{MIN_DESCRIPTION_LENGTH} chars): "
        f"{short_descriptions}"
    )


async def test_every_tool_has_an_input_schema(mcp_with_tools: FastMCP) -> None:
    """Pydantic input models must produce a JSON schema the LLM can read."""
    tools = await mcp_with_tools.list_tools()

    for tool in tools:
        assert tool.inputSchema, f"{tool.name} has no inputSchema"
        assert tool.inputSchema.get("type") == "object", (
            f"{tool.name} inputSchema is not an object: {tool.inputSchema}"
        )


@pytest.mark.parametrize(
    "tool_name, keyword",
    [
        ("get_apod_tool", "title"),
        ("search_apod_tool", "explanation"),
        ("get_epic_images_tool", "jpg_url"),
        ("get_epic_available_dates_tool", "YYYY-MM-DD"),
        ("get_rover_photos_tool", "img_src"),
        ("get_rover_manifest_tool", "photo_manifest"),
        ("get_neo_feed_tool", "close_approach_data"),
        ("get_neo_lookup_tool", "orbital_data"),
    ],
)
async def test_tool_description_mentions_response_field(
    mcp_with_tools: FastMCP, tool_name: str, keyword: str
) -> None:
    """Each tool's description names at least one field from its response shape."""
    tools = await mcp_with_tools.list_tools()
    tool = next((t for t in tools if t.name == tool_name), None)
    assert tool is not None, f"{tool_name} not registered"
    assert keyword in (tool.description or ""), (
        f"{tool_name} description should mention `{keyword}` (response shape) "
        f"but doesn't:\n{tool.description}"
    )
