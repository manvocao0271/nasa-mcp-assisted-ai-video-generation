"""Thin wrapper around the Qwen Cloud chat API (OpenAI-compatible).

Qwen Cloud exposes a DashScope endpoint that is fully compatible with the OpenAI Python SDK. All agents use this client for LLM calls.

Docs: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

Base URL:  https://dashscope-intl.aliyuncs.com/compatible-mode/v1
Auth:      QWEN_API_KEY (set in .env / environment)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
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

# Chat (Streamlit ChatAgent) — separate from data/vision to allow fine-tuned overrides
MODEL_CHAT = os.getenv("QWEN_CHAT_MODEL", "qwen3.7-plus")


def _load_qwen_api_keys(fallback: str = "") -> list[str]:
    """Load QWEN_API_KEY_0, QWEN_API_KEY_1, QWEN_API_KEY_2, ... from the
    environment, in order, stopping at the first missing index.

    Canonical implementation — shared by QwenClient (this module, used by
    DataAgent/ScriptAgent/StoryboardAgent/ChatAgent) and VideoGen
    (agent/video_gen.py, which imports this function rather than keeping
    its own copy), so multi-account cycling behaves identically everywhere
    and there's exactly one place to fix if that behavior ever needs to
    change.

    If no numbered keys are set, falls back to a single key — either the
    bare QWEN_API_KEY env var, or whatever the caller passed in directly —
    so existing single-key setups need zero .env changes to keep working.
    """
    keys: list[str] = []
    i = 0
    while True:
        k = os.environ.get(f"QWEN_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1

    if not keys:
        bare = os.environ.get("QWEN_API_KEY", "").strip() or fallback.strip()
        if bare:
            keys = [bare]

    # Dedupe while preserving order, in case QWEN_API_KEY and QWEN_API_KEY_0
    # both happen to be set to the same value.
    seen: set[str] = set()
    deduped: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _is_cyclable_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a per-account-blocked error that a
    different Qwen Cloud account's key might succeed past — free-tier quota
    exhaustion, or that specific key lacking model-scope authorization.

    Both surface as HTTP 403 from DashScope. Deliberately broader than just
    literal quota exhaustion: since different accounts can have different
    per-model authorization scopes (as seen firsthand configuring this
    project's own API key), an access-denied 403 is just as plausibly
    solved by trying a different account as a quota-exhausted one is.

    Does NOT trigger on model-not-found errors — those indicate the model
    string itself is wrong or unregistered, which is a project-wide
    misconfiguration that will fail identically on every account, not
    something switching keys can fix.
    """
    status = getattr(exc, "status_code", None)
    if status == 403:
        return True
    # Fallback for wrapped/transport-level exceptions that might not expose
    # a clean status_code attribute — same substrings VideoGen checks for.
    msg = str(exc)
    return "AllocationQuota" in msg or "FreeTier" in msg or "access_denied" in msg


def _log_api_call(key: str, model: str, latency: float, status: str, request_id: str = "") -> None:
    """Print one terminal log line per real Qwen Cloud API call — mirrors
    the columns in DashScope's own console request log (Timestamp, Model,
    API Key, Latency, Status, Request ID). Usage/token counts are
    deliberately not logged here, matching what was asked for.

    Key display: DashScope's console shows its own short numeric ID for a
    key (e.g. "336906") — that's a dashboard-side identifier, not something
    any API response returns to the client, so it can't be reproduced here.
    Instead this shows a short, safe, locally-derived suffix of the actual
    key (its last 6 characters) — enough to tell which configured key was
    used without ever printing a full secret to the terminal.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    key_display = f"...{key[-6:]}" if len(key) >= 6 else (key or "unknown")
    req_display = request_id or "unknown"
    print(f"[Qwen] {ts} | key {key_display} | {model} | {latency:.2f}s | {status} | req={req_display}")


class QwenClient:
    """Synchronous Qwen Cloud chat client with token-usage tracking.

    Note: import-time side effects are minimal — the heavy `openai` import is
    deferred until an instance is constructed so top-level imports can be used
    in lightweight tooling without installing `openai`.

    Multi-account cycling: if QWEN_API_KEY_0 / QWEN_API_KEY_1 / ... are set,
    every call automatically cycles through them on a 403 (quota exhausted or
    access denied) before giving up — same behavior/rationale as VideoGen's
    I2V/T2V model fallback chains, applied here to text/vision calls instead.
    A single QWEN_API_KEY (or a key passed in directly) still works exactly
    as before if no numbered keys are configured.
    """

    def __init__(self, api_key: str, model: str = MODEL_PLUS) -> None:
        # Deferred import so merely importing this module doesn't require `openai`.
        from openai import OpenAI  # type: ignore

        self.model = model
        self.tokens_used = 0
        self._api_keys = _load_qwen_api_keys(fallback=api_key)
        if not self._api_keys:
            raise ValueError(
                "No Qwen Cloud API key available. Set QWEN_API_KEY, or "
                "QWEN_API_KEY_0 / QWEN_API_KEY_1 / ... in .env for "
                "multi-account cycling."
            )
        self._clients = [OpenAI(api_key=k, base_url=QWEN_BASE_URL) for k in self._api_keys]
        # Which key/client index succeeded most recently — each new call
        # starts cycling from here rather than always index 0, so a
        # multi-call sequence (e.g. DataAgent's tool-calling loop, which
        # reuses one QwenClient instance across several .chat() calls)
        # doesn't keep re-attempting an already-known-exhausted key every
        # single time.
        self._current_key_idx = 0

    def _call_with_key_cycling(self, make_request):
        """Call *make_request(client)* against each configured client,
        starting at self._current_key_idx and wrapping around, cycling to
        the next one only on a 403-class error (see _is_cyclable_error).
        Any other exception propagates immediately without cycling — an
        identical non-403 failure (bad request, model not found, etc.)
        would fail the same way on every account.

        Every real attempt (success or failure) is logged via
        _log_api_call — this is the single chokepoint all chat/vision
        calls pass through, so logging here covers everything once
        rather than duplicating it in chat()/stream() separately.
        """
        n = len(self._clients)
        last_exc: Exception | None = None
        for offset in range(n):
            idx = (self._current_key_idx + offset) % n
            _t0 = time.monotonic()
            try:
                result = make_request(self._clients[idx])
                _latency = time.monotonic() - _t0
                _req_id = getattr(result, "_request_id", "") or ""
                _log_api_call(self._api_keys[idx], self.model, _latency, "200", _req_id)
                self._current_key_idx = idx
                return result
            except Exception as e:  # noqa: BLE001 - intentionally broad, see docstring
                _latency = time.monotonic() - _t0
                _status = str(getattr(e, "status_code", None) or "ERR")
                _req_id = getattr(e, "request_id", "") or ""
                _log_api_call(self._api_keys[idx], self.model, _latency, _status, _req_id)
                last_exc = e
                if not _is_cyclable_error(e):
                    raise
                if n > 1:
                    print(f"[QwenClient] key #{idx} blocked (403) on model '{self.model}' — trying next key")
                continue
        # Every configured key failed with a cyclable (403) error.
        assert last_exc is not None
        raise last_exc

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Send a chat request and return the full completion object.

        Builds the kwargs dict and calls the Qwen Cloud chat completions API,
        cycling across configured API keys on a 403 (see class docstring).
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self._call_with_key_cycling(
                lambda client: client.chat.completions.create(**kwargs)
            )
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

    def stream(self, messages: list[dict]) -> Any:
        """Open a streaming chat completion and return the raw stream iterator.

        Yields openai ``ChatCompletionChunk`` objects.  Callers should read
        ``chunk.choices[0].delta.content`` for text tokens.

        Key-cycling applies only to the initial request that opens the
        stream (a 403 there cycles keys same as chat()) — a failure
        partway through an already-open stream is not retried, since a
        partially-consumed stream can't be cleanly resumed on another key.
        """
        return self._call_with_key_cycling(
            lambda client: client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )
        )

    def chat_json(self, messages: list[dict]) -> dict:
        """Call Qwen and parse the response content as JSON.

        Uses response_format=json_object to guarantee valid JSON output.
        """
        response = self.chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content or "{}"
        return json.loads(content)