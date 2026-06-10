"""Unit tests for DONKI input models, API clients, and MCP tool registration."""

from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from nasa_mcp.cache import Cache
from nasa_mcp.config import Config
from nasa_mcp.errors import NasaApiError
from nasa_mcp.features.donki import register_donki_tools
from nasa_mcp.features.donki.cme import api as cme_api
from nasa_mcp.features.donki.cme.inputs import GetCMEEventsInput
from nasa_mcp.features.donki.flr import api as flr_api
from nasa_mcp.features.donki.flr.inputs import GetFLREventsInput
from nasa_mcp.features.donki.gst import api as gst_api
from nasa_mcp.features.donki.gst.inputs import GetGSTEventsInput

DonkiApiFunc = Callable[[Config, date | None, date | None], Awaitable[list[dict]]]


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    return Config(
        nasa_api_key="TEST_KEY",
        cache_path=tmp_path / "donki-cache.sqlite3",
        request_timeout=5.0,
    )


@pytest.mark.parametrize(
    "input_model",
    [
        GetCMEEventsInput,
        GetFLREventsInput,
        GetGSTEventsInput,
    ],
)
def test_donki_inputs_allow_omitted_dates(input_model: type) -> None:
    args = input_model()

    assert args.start_date is None
    assert args.end_date is None


@pytest.mark.parametrize(
    "input_model",
    [
        GetCMEEventsInput,
        GetFLREventsInput,
        GetGSTEventsInput,
    ],
)
def test_donki_inputs_accept_inclusive_date_range(input_model: type) -> None:
    args = input_model(start_date=date(2024, 5, 10), end_date=date(2024, 5, 10))

    assert args.start_date == date(2024, 5, 10)
    assert args.end_date == date(2024, 5, 10)


@pytest.mark.parametrize(
    "input_model",
    [
        GetCMEEventsInput,
        GetFLREventsInput,
        GetGSTEventsInput,
    ],
)
def test_donki_inputs_reject_partial_date_range(input_model: type) -> None:
    with pytest.raises(Exception, match="Provide both start_date and end_date"):
        input_model(start_date=date(2024, 5, 10))


@pytest.mark.parametrize(
    "input_model",
    [
        GetCMEEventsInput,
        GetFLREventsInput,
        GetGSTEventsInput,
    ],
)
def test_donki_inputs_reject_reversed_date_range(input_model: type) -> None:
    with pytest.raises(Exception, match="end_date must be on or after start_date"):
        input_model(start_date=date(2024, 5, 11), end_date=date(2024, 5, 10))


@pytest.mark.parametrize(
    "api_module,api_func,endpoint",
    [
        (cme_api, cme_api.get_cme_events, "https://api.nasa.gov/DONKI/CME"),
        (flr_api, flr_api.get_flr_events, "https://api.nasa.gov/DONKI/FLR"),
        (gst_api, gst_api.get_gst_events, "https://api.nasa.gov/DONKI/GST"),
    ],
)
async def test_donki_api_uses_expected_endpoint_and_params(
    monkeypatch: pytest.MonkeyPatch,
    test_config: Config,
    api_module: Any,
    api_func: DonkiApiFunc,
    endpoint: str,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def get(self, url: str, params: dict[str, str]) -> SimpleNamespace:
            calls.append((url, params))
            return SimpleNamespace(
                status_code=200,
                json=lambda: [{"activityID": "2024-05-10T00:00:00-DONKI-001"}],
                text="OK",
            )

    monkeypatch.setattr(api_module.httpx, "AsyncClient", FakeAsyncClient)

    result = await api_func(test_config, date(2024, 5, 10), date(2024, 5, 11))

    assert result == [{"activityID": "2024-05-10T00:00:00-DONKI-001"}]
    assert calls == [
        (
            endpoint,
            {
                "api_key": "TEST_KEY",
                "startDate": "2024-05-10",
                "endDate": "2024-05-11",
            },
        )
    ]


@pytest.mark.parametrize(
    "api_func,error_match",
    [
        (cme_api.get_cme_events, "DONKI-CME requires both"),
        (flr_api.get_flr_events, "DONKI-FLR requires both"),
        (gst_api.get_gst_events, "DONKI-GST requires both"),
    ],
)
async def test_donki_api_rejects_partial_date_range(
    test_config: Config,
    api_func: DonkiApiFunc,
    error_match: str,
) -> None:
    with pytest.raises(NasaApiError, match=error_match):
        await api_func(test_config, date(2024, 5, 10), None)


async def test_register_donki_tools_registers_all_subfeature_tools(test_config: Config, tmp_path: Path) -> None:
    cache = Cache(tmp_path / "donki-tools-cache.sqlite3")
    mcp = FastMCP("donki-test")

    register_donki_tools(mcp, test_config, cache)

    tools = await mcp.list_tools()
    names = {tool.name for tool in tools}

    assert {
        "get_cme_events_tool",
        "get_flr_events_tool",
        "get_gst_events_tool",
    } <= names


@pytest.mark.parametrize(
    "tool_name,keyword",
    [
        ("get_cme_events_tool", "cmeAnalyses"),
        ("get_flr_events_tool", "class"),
        ("get_gst_events_tool", "Kp"),
    ],
)
async def test_donki_tool_descriptions_are_substantive(
    test_config: Config,
    tmp_path: Path,
    tool_name: str,
    keyword: str,
) -> None:
    cache = Cache(tmp_path / f"{tool_name}.sqlite3")
    mcp = FastMCP("donki-description-test")

    register_donki_tools(mcp, test_config, cache)
    tools = await mcp.list_tools()
    tool = next((tool for tool in tools if tool.name == tool_name), None)

    assert tool is not None
    assert tool.description is not None
    assert len(tool.description) >= 150
    assert keyword in tool.description
