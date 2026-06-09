"""Smoke test for the InSight Mars Weather scaffold."""

from nasa_mcp.features.insight.tools import register_insight_tools


def test_register_insight_tools_is_callable() -> None:
    assert callable(register_insight_tools)
