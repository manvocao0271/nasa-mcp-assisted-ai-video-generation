"""Unit tests for APOD input models."""

from datetime import date

from nasa_mcp.features.apod.inputs import GetApodInput, SearchApodInput


def test_get_apod_input_accepts_optional_date() -> None:
    args = GetApodInput(target_date=date(2024, 1, 1))

    assert args.target_date == date(2024, 1, 1)


def test_get_apod_input_allows_omitted_date() -> None:
    args = GetApodInput()

    assert args.target_date is None


def test_search_apod_input_accepts_query_and_optional_range() -> None:
    args = SearchApodInput(
        query="black hole",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    assert args.query == "black hole"
    assert args.start_date == date(2024, 1, 1)
    assert args.end_date == date(2024, 1, 31)
