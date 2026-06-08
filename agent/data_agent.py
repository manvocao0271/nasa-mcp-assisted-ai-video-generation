"""Data Agent — translates the user request into NASA data assets via MCP tools.

Spins up the nasa-mcp MCP server as a stdio subprocess, hands the user message
to Qwen (qwen-max) in tool-calling mode, executes the resulting MCP tool calls,
and returns a structured assets dict saved to output/assets.json.

Output schema (output/assets.json):
    {
        "query": str,               # original user message
        "tools_called": [str],      # MCP tool names used
        "images": [                 # NASA images suitable as Wan reference frames
            {"url": str, "caption": str, "source": str}
        ],
        "data": {                   # raw scientific data keyed by tool name
            "<tool_name>": <result>
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from openai.types.chat import ChatCompletionMessageToolCall

from agent.qwen_client import MODEL_MAX, QwenClient

OUTPUT_DIR = Path("output")
MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are a NASA data researcher. Use the available NASA tools to gather images, \
scientific data, and context relevant to the user's request. \
Call multiple tools to collect rich, varied assets — aim for at least 2-3 tool calls. \
When you have enough data to write a compelling short film, stop calling tools and \
summarise what you found in one sentence.\
"""


def _unwrap_args_schema(schema: dict) -> dict:
    """FastMCP wraps all tool parameters under a single 'args' property.

    Unwrap it so Qwen sees the actual parameter schema directly.
    Carries over any $defs from the outer schema so $ref links stay valid.
    """
    props = schema.get("properties", {})
    if list(props.keys()) == ["args"] and isinstance(props["args"], dict):
        inner = dict(props["args"])
        if "$defs" in schema:
            inner["$defs"] = schema["$defs"]
        return inner
    return schema


class DataAgent:
    """Calls NASA MCP tools via an agentic Qwen loop and collects raw assets."""

    def __init__(self, nasa_api_key: str, qwen_api_key: str) -> None:
        self.nasa_api_key = nasa_api_key
        self.qwen_api_key = qwen_api_key

    def run(self, user_message: str) -> dict:
        """Fetch NASA data relevant to the user message.

        Runs the async MCP loop synchronously and writes output/assets.json.
        Returns the assets dict.
        """
        assets = asyncio.run(self._fetch_assets(user_message))
        (OUTPUT_DIR / "assets.json").write_text(json.dumps(assets, indent=2))
        return assets

    async def _fetch_assets(self, user_message: str) -> dict:
        """Async: spin up the MCP server and run the tool-calling loop."""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "nasa-mcp"],
            env={**os.environ, "NASA_API_KEY": self.nasa_api_key},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools_result = await session.list_tools()

                # Build OpenAI-compatible tool schemas from MCP tool descriptors
                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": _unwrap_args_schema(
                                tool.inputSchema
                                if isinstance(tool.inputSchema, dict)
                                else {}
                            ),
                        },
                    }
                    for tool in mcp_tools_result.tools
                ]

                client = QwenClient(self.qwen_api_key, model=MODEL_MAX)
                messages: list[dict] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                tools_called: list[str] = []
                all_results: list[dict] = []

                for _ in range(MAX_TOOL_ITERATIONS):
                    response = client.chat(messages, tools=openai_tools)
                    msg = response.choices[0].message

                    # Append assistant turn (tool_calls or final text)
                    messages.append(msg.model_dump(exclude_none=True))

                    if not msg.tool_calls:
                        break  # Qwen is done calling tools

                    for tc in msg.tool_calls:
                        if not isinstance(tc, ChatCompletionMessageToolCall):
                            continue
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments or "{}")
                        tools_called.append(tool_name)

                        # Call the MCP tool — FastMCP expects args wrapped under "args"
                        mcp_result = await session.call_tool(
                            tool_name, {"args": tool_args}
                        )
                        result_text = next(
                            (item.text for item in mcp_result.content if isinstance(item, TextContent)),
                            "{}",
                        )

                        try:
                            parsed = json.loads(result_text)
                        except json.JSONDecodeError:
                            parsed = {"raw": result_text}

                        all_results.append({"tool": tool_name, "result": parsed})

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            }
                        )

                return self._build_assets(user_message, tools_called, all_results)

    def _build_assets(
        self, query: str, tools_called: list[str], results: list[dict]
    ) -> dict:
        """Distil raw tool results into the assets dict consumed by downstream agents."""
        images: list[dict] = []
        data: dict = {}

        IMAGE_URL_KEYS = ("hdurl", "url", "jpg_url", "img_src")

        for entry in results:
            tool = entry["tool"]
            result = entry["result"]

            if isinstance(result, dict):
                for key in IMAGE_URL_KEYS:
                    if key in result and isinstance(result[key], str):
                        images.append(
                            {
                                "url": result[key],
                                "caption": (
                                    result.get("title")
                                    or str(result.get("explanation", ""))[:120]
                                ),
                                "source": tool,
                            }
                        )
                        break
                data[tool] = result

            elif isinstance(result, list):
                for item in result[:3]:
                    if isinstance(item, dict):
                        for key in IMAGE_URL_KEYS:
                            if key in item and isinstance(item[key], str):
                                images.append(
                                    {
                                        "url": item[key],
                                        "caption": (
                                            item.get("title")
                                            or str(item.get("explanation", ""))[:120]
                                        ),
                                        "source": tool,
                                    }
                                )
                                break
                data[tool] = result

        return {
            "query": query,
            "tools_called": tools_called,
            "images": images[:5],  # cap at 5 reference frames for Wan
            "data": data,
        }
