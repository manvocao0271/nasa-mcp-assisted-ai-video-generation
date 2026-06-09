"""Smoke test for the EONET scaffold."""

from nasa_mcp.features.eonet.tools import register_eonet_tools


def test_register_eonet_tools_is_callable() -> None:
    assert callable(register_eonet_tools)
