"""Smoke test for the Mars Trek WMTS scaffold."""

from nasa_mcp.features.mars_trek.tools import register_mars_trek_tools


def test_register_mars_trek_tools_is_callable() -> None:
    assert callable(register_mars_trek_tools)
