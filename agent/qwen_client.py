"""Thin wrapper around the Qwen Cloud chat API (OpenAI-compatible).

Qwen Cloud exposes a DashScope endpoint that is fully compatible with the OpenAI Python SDK. All agents use this client for LLM calls.

Docs: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

Base URL:  https://dashscope-intl.aliyuncs.com/compatible-mode/v1
Auth:      QWEN_API_KEY (set in .env / environment)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checkers; avoid runtime import errors when `openai`
    # isn't installed in lightweight environments used for quick checks.
    from openai.types.chat import ChatCompletion

def _load_project_dotenv(env_filename: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from the project `.env` without overriding
    any variables already present in `os.environ`.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / env_filename
        if not env_path.exists():
            return
        text = env_path.read_text(encoding="utf8")
    except Exception:
        return

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        else:
            # drop inline comments after an unquoted value
            if "#" in val:
                val = val.split("#", 1)[0].rstrip()
        # Set/override env vars from project .env so the project file takes precedence.
        if key:
            os.environ[key] = val


# Load project .env early so subsequent env lookups use its values.
_load_project_dotenv()

QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Text + tool-calling (Data Agent) — lighter than Max to preserve budget for video gen
MODEL_PLUS = os.getenv("QWEN_MODEL_DATA", "qwen3.7-plus")

# Vision (Script + Storyboard) — VL tier with free-quota on Qwen Cloud
MODEL_VL_PLUS = os.getenv("QWEN_MODEL_VISION", "qwen-vl-plus-2024-08-13")


class QwenClient:
    """Synchronous Qwen Cloud chat client with token-usage tracking.

    Note: import-time side effects are minimal — the heavy `openai` import is
    deferred until an instance is constructed so top-level imports can be used
    in lightweight tooling without installing `openai`.
    """

    def __init__(self, api_key: str, model: str = MODEL_PLUS) -> None:
        # Deferred import so merely importing this module doesn't require `openai`.
        from openai import OpenAI  # type: ignore

        self.model = model
        self.tokens_used = 0
        self._client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Send a chat request and return the full completion object.

        Builds the kwargs dict and calls the Qwen Cloud chat completions API.
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if "model" in msg and ("does not exist" in msg or "model_not_found" in msg):
                raise RuntimeError(
                    f"Qwen model '{self.model}' not found or you do not have access to it.\n"
                    "Set the `QWEN_MODEL_VISION` (or `QWEN_MODEL_DATA`) environment variable to a model ID\n"
                    "that your QWEN account can use, or request access to the model in the Qwen Cloud dashboard.\n"
                    "To list models available to your API key, run:\n"
                    "python -c \"from openai import OpenAI; import os; client=OpenAI(api_key=os.environ['QWEN_API_KEY'], base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'); print([m.id for m in client.models.list().data])\""
                ) from e
            raise

        if getattr(response, "usage", None):
            try:
                self.tokens_used += response.usage.total_tokens
            except Exception:
                pass
        return response

    def chat_json(self, messages: list[dict]) -> dict:
        """Call Qwen and parse the response content as JSON.

        Uses response_format=json_object to guarantee valid JSON output.
        """
        response = self.chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
