"""Unit tests for NeoWs input models."""

from datetime import date

from nasa_mcp.features.neo.inputs import GetNeoFeedInput, GetNeoLookupInput
def test_get_neo_feed_input_accepts_optional_date() -> None:
    args = GetNeoFeedInput(start_date=date(2026, 5, 31), end_date=date(2026, 6, 7))

    assert args.end_date == date(2026, 6, 7)


def test_get_neo_feed_input_allows_ommitted_date() -> None:
    args = GetNeoFeedInput(start_date=date(2026, 5, 31))

    assert args.end_date is None


def test_get_neo_lookup_input_accepts_asteroid_id() -> None:
    args = GetNeoLookupInput(asteroid_id="3542519")

    assert args.asteroid_id == "3542519"