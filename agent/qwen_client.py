"""Thin wrapper around the Qwen Cloud chat API (OpenAI-compatible).

Qwen Cloud exposes a DashScope endpoint that is fully compatible with the OpenAI Python SDK. All agents use this client for LLM calls.

Docs: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

Base URL:  https://dashscope-intl.aliyuncs.com/compatible-mode/v1
Auth:      QWEN_API_KEY (set in .env / environment)
"""

from __future__ import annotations

import json
import os

from openai import OpenAI
from openai.types.chat import ChatCompletion

QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Text + tool-calling (Data Agent) — lighter than Max to preserve budget for video gen
MODEL_PLUS = os.getenv("QWEN_MODEL_DATA", "qwen3.7-plus")

# Vision (Script + Storyboard) — VL tier with free-quota on Qwen Cloud
MODEL_VL_PLUS = os.getenv("QWEN_MODEL_VISION", "qwen-vl-plus-2024-08-13")


class QwenClient:
    """Synchronous Qwen Cloud chat client with token-usage tracking."""

    def __init__(self, api_key: str, model: str = MODEL_PLUS) -> None:
        self.model = model
        self.tokens_used = 0
        self._client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ChatCompletion:
        """Send a chat request and return the full completion object.

        Tracks cumulative token usage in self.tokens_used.
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        if response.usage:
            self.tokens_used += response.usage.total_tokens
        return response

    def chat_json(self, messages: list[dict]) -> dict:
        """Call Qwen and parse the response content as JSON.

        Uses response_format=json_object to guarantee valid JSON output.
        """
        response = self.chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
