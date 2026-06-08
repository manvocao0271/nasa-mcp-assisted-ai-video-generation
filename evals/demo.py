"""End-to-end demo: Groq LLM driving the nasa-mcp server via MCP stdio.

Launches the nasa-mcp server as a subprocess, hands all registered tools to Groq, then runs a conversational agentic loop for each question. Prints the tool-call trace and final answer for every query.

Usage:
    GROQ_API_KEY=gsk_... uv run python evals/demo.py
    GROQ_API_KEY=gsk_... NASA_API_KEY=your_key uv run python evals/demo.py

The NASA_API_KEY defaults to DEMO_KEY (rate-limited). Set it for reliable runs. Groq's free tier supports llama-3.3-70b-versatile with tool calling.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
MODEL = "llama-3.3-70b-versatile"

QUESTIONS = [
    "What was the Astronomy Picture of the Day on July 20, 1969 (Apollo 11 moon landing)?",
    "What was the Astronomy Picture of the Day on Barack Obama's inauguration?",
    "Show me the most recent full-disc Earth image taken from space by DSCOVR's EPIC camera.",
    "Are there any potentially hazardous asteroids approaching Earth in the next 7 days starting from 2026-06-07?",
]


def _unwrap_args_schema(schema: dict) -> dict:
    """FastMCP nests every tool's parameters under a single 'args' property.
    Unwrap it so the LLM sees the inner schema directly.
    Carries over any $defs from the outer schema so $ref links stay valid.
    """

    props = schema.get("properties", {})
    if list(props.keys()) == ["args"] and isinstance(props["args"], dict):
        inner = dict(props["args"])
        if "$defs" in schema:
            inner["$defs"] = schema["$defs"]
        return inner
    return schema


def _needs_args_wrap(schema: dict) -> bool:
    """Return True if the original MCP schema uses the FastMCP 'args' wrapper."""

    props = schema.get("properties", {})
    return list(props.keys()) == ["args"]


def mcp_tools_to_openai(tools: list) -> tuple[list[dict], dict[str, bool]]:
    """Convert MCP tool definitions to OpenAI function-calling format.

    Returns (openai_tools, wrap_map) where wrap_map[tool_name] is True when
    the tool's arguments must be re-wrapped in {'args': ...} before calling MCP.
    """

    openai_tools = []
    wrap_map: dict[str, bool] = {}
    for tool in tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        wrap_map[tool.name] = _needs_args_wrap(schema)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _unwrap_args_schema(schema),
                },
            }
        )
    return openai_tools, wrap_map


async def run_agentic_loop(
    openai: AsyncOpenAI,
    session: ClientSession,
    openai_tools: list[dict],
    wrap_map: dict[str, bool],
    question: str,
) -> None:
    """Run a single question through the full LLM → tool → LLM loop."""

    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a helpful NASA data assistant. Use the available tools to answer questions with real data. Be concise but include key facts, numbers, and URLs from the tool responses. APOD only has data from 1995-06-16 onwards — do not call it for earlier dates. If a tool returns an error, explain the limitation and stop retrying."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Agentic loop — keep calling tools until the model stops requesting them
    while True:
        response = await openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,  # type: ignore[arg-type]
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        msg = response.choices[0].message

        # Add assistant turn to history
        messages.append(msg.model_dump(exclude_unset=True))  # type: ignore[arg-type]

        if not msg.tool_calls:
            # Model is done — print final answer
            print(f"\nANSWER:\n{msg.content}")
            break

        # Execute each requested tool call via the MCP server
        for tool_call in msg.tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                continue
            fn = tool_call.function
            args = json.loads(fn.arguments) if fn.arguments else {}

            # Re-wrap in {'args': ...} if the MCP tool expects the FastMCP wrapper
            mcp_args = {"args": args} if wrap_map.get(fn.name) else args

            print(f"\n  → Tool call : {fn.name}")
            print(f"    Arguments : {json.dumps(args, indent=6)}")

            result = await session.call_tool(fn.name, arguments=mcp_args)

            # MCP returns a list of content blocks; extract text
            content_text = "\n".join(
                block.text
                for block in result.content
                if isinstance(block, TextContent)
            )

            # Truncate before sending back to the model to stay within TPM limits.
            # Keep enough for the model to extract key facts.
            MAX_TOOL_CHARS = 4000
            model_content = (
                content_text[:MAX_TOOL_CHARS] + "\n[...truncated]"
                if len(content_text) > MAX_TOOL_CHARS
                else content_text
            )

            # Shorter preview for the printed trace
            preview = content_text[:300] + "..." if len(content_text) > 300 else content_text
            print(f"    Result    : {preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": model_content,
                }
            )


async def main() -> None:
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set.", file=sys.stderr)
        print("Run: GROQ_API_KEY=gsk_... uv run python evals/demo.py", file=sys.stderr)
        sys.exit(1)

    openai = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    # Resolve the server entry point relative to this file
    server_path = Path(__file__).parent.parent / "nasa_mcp" / "server.py"

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", str(server_path)],
        env={**os.environ, "NASA_API_KEY": NASA_API_KEY},
    )

    print(f"Starting nasa-mcp server  (NASA_API_KEY={'set' if NASA_API_KEY != 'DEMO_KEY' else 'DEMO_KEY'})")
    print(f"Using Groq model          : {MODEL}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            print(f"Registered tools ({len(mcp_tools)}):")
            for t in mcp_tools:
                print(f"  • {t.name}")

            openai_tools, wrap_map = mcp_tools_to_openai(mcp_tools)

            for question in QUESTIONS[1:2]:
                try:
                    await run_agentic_loop(openai, session, openai_tools, wrap_map, question)
                except Exception as exc:
                    print(f"\n  [ERROR] Question skipped: {exc}")

    print(f"\n{'=' * 70}")
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
