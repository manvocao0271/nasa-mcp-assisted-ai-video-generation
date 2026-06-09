"""Unit tests for NASA Exoplanet Archive input models."""

import pytest

from nasa_mcp.features.exoplanets.inputs import SearchExoplanetsInput


def test_search_exoplanets_input_defaults() -> None:
    args = SearchExoplanetsInput(query="TRAPPIST")
    assert args.query == "TRAPPIST"
    assert args.limit == 20


def test_search_exoplanets_input_accepts_limit() -> None:
    args = SearchExoplanetsInput(query="Kepler", limit=50)
    assert args.limit == 50


def test_search_exoplanets_input_rejects_zero_limit() -> None:
    with pytest.raises(Exception):
        SearchExoplanetsInput(query="test", limit=0)


def test_search_exoplanets_input_rejects_limit_above_200() -> None:
    with pytest.raises(Exception):
        SearchExoplanetsInput(query="test", limit=201)
