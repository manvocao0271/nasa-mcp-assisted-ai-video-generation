"""Chat agent — multi-turn conversation with optional NASA asset grounding.

Handles user inquiries about the universe with persistent conversation history.
Optionally triggers video generation when the user explicitly asks for a video.
"""

from __future__ import annotations

import os
import re
from datetime import date

from agent.qwen_client import QwenClient
from agent.retriever import Retriever


def _build_system_prompt() -> str:
    today = date.today().strftime("%B %d, %Y")
    return f"""You are WILL.ai (What Infinity Looks Like AI), an expert astronomy guide and universe educator backed by live NASA data.
Today's date is {today}.

When NASA data is provided below in a "Retrieved NASA data" block, treat it as ground truth
and cite specific facts, titles, dates, or descriptions from it in your answer.

When answering:
1. Explain concepts clearly and concisely
2. Use any provided NASA data as grounding — prefer it over your training knowledge for current events
3. If the system indicates NASA search returned no results, say so clearly.

Do not make up APOD titles, dates, or image descriptions — only state what is in the provided data.
"""


SYSTEM_PROMPT = _build_system_prompt()

# Keywords that indicate the user wants live NASA data
_LIVE_DATA_PATTERNS = (
    r"\bapod\b",
    r"\bastronomy picture of the day\b",
    r"\bpicture of the day\b",
    r"\btoday'?s? (image|picture|photo|astronomy)\b",
    r"\byesterday'?s? (image|picture|photo|astronomy)\b",
    r"\blatest (image|picture|photo|news|discovery)\b",
    r"\brecent (solar flare|cme|asteroid|comet|discovery)\b",
    r"\bnear.?earth asteroid\b",
    r"\bsolar flare\b",
    r"\bcoronal mass ejection\b",
    r"\bcme\b",
    r"\bgeomagnetic storm\b",
    r"\bepic (image|earth|photo)\b",
    r"\bearth (today|yesterday|right now|from space)\b",
    r"\bwhat does .+ look like (today|right now|currently)\b",
    r"\bshow me .+nasa\b",
    # General visual / image request patterns
    r"\bshow me (a |an |some )?(pictures?|photos?|images?)\b",
    r"\bshow me what .+ looks? like\b",
    r"\bwhat does .+ look like\b",
    r"\b(pictures?|photos?|images?) of\b",
    r"\b(find|get|fetch|retrieve|search for) .*(pictures?|photos?|images?)\b",
    r"\bnasa (pictures?|photos?|images?|resources?)\b",
    r"\bcan you (show|find|fetch|get|retrieve).*(picture|photo|image)\b",
)


def _needs_live_data(text: str) -> bool:
    """Return True if the query is best answered with a live NASA MCP fetch."""
    t = text.lower()
    return any(re.search(p, t) for p in _LIVE_DATA_PATTERNS)


_IMAGE_URL_RE = re.compile(
    r"https?://\S+\.(?:jpg|jpeg|png|gif|webp)(?:\?\S*)?",
    re.IGNORECASE,
)


def _extract_image_urls(text: str) -> list[str]:
    """Return any direct image URLs found in the user's message."""
    return _IMAGE_URL_RE.findall(text)


def _assets_from_urls(urls: list[str], query: str) -> dict:
    """Build a minimal assets dict from user-supplied image URLs."""
    images = [
        {"url": url, "thumb_url": "", "caption": "", "source": "user_provided"}
        for url in urls
    ]
    return {"query": query, "tools_called": [], "images": images, "data": {}}





class ChatAgent:
    """Maintains multi-turn conversations with astronomy context."""

    def __init__(self, qwen_client: QwenClient):
        self.client = qwen_client

    def answer(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """Answer a user query with conversation context."""
        if conversation_history is None:
            conversation_history = []

        messages = [{"role": "system", "content": _build_system_prompt()}]

        passages = []
        chat_assets: dict = {}
        user_image_urls = _extract_image_urls(user_message)
        if user_image_urls:
            chat_assets = _assets_from_urls(user_image_urls, user_message)
        elif _needs_live_data(user_message):
            try:
                from agent.data_agent import DataAgent
                nasa_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
                qwen_key = os.environ.get("QWEN_API_KEY", "")
                chat_assets = DataAgent(nasa_key, qwen_key).fetch(user_message)
                retriever = Retriever()
                passages = retriever._extract_passages(chat_assets, top_k=6)
                if passages:
                    grounding = retriever.format_for_prompt(passages)
                    messages.append({"role": "system", "content": grounding})
                else:
                    messages.append({
                        "role": "system",
                        "content": (
                            "A NASA image/data search was performed for this query but "
                            "returned no results. Tell the user that NASA's public databases "
                            "did not return images matching their request. Do not invent "
                            "or describe images you do not have."
                        ),
                    })
            except Exception:
                passages = []
                chat_assets = {}

        # Use simple user/assistant pairs — the messages field stores the full
        # accumulated history per run, so appending all runs' messages lists
        # causes exponential duplication that floods the context window.
        for prev_run in conversation_history[-5:]:
            messages.append({"role": "user", "content": prev_run["user_message"]})
            messages.append({"role": "assistant", "content": prev_run["assistant_response"]})

        messages.append({"role": "user", "content": user_message})

        response = self.client.chat(messages=messages)
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "retrieved_passages": passages,
            "chat_assets": chat_assets,
        }

    def answer_stream_internal(
        self,
        user_message: str,
        conversation_history: list[dict] | None,
        result: dict,
    ):
        """Yield plain-text tokens and populate *result* when done.

        Designed for use with ``st.write_stream()``:

            result = {}
            answer = st.write_stream(
                chat_agent.answer_stream_internal(msg, history, result)
            )

        After ``st.write_stream`` returns, *result* contains:
        - ``answer``               – full response string
        - ``retrieved_passages``   – list[dict]
        - ``chat_assets``          – NASA images/data fetched for this turn

        Thinking tokens enclosed in ``<think>…</think>`` are filtered out so
        they never reach the UI, even when a reasoning-capable model is used.
        """
        if conversation_history is None:
            conversation_history = []

        messages = [{"role": "system", "content": _build_system_prompt()}]

        # For live-data queries (APOD, solar flares, NEO, etc.) fetch from NASA MCP
        # without writing to disk, then inject as grounding context.
        passages: list = []
        chat_assets: dict = {}

        # Priority 1: user pasted direct image URL(s) — use them as-is, skip MCP fetch
        user_image_urls = _extract_image_urls(user_message)
        if user_image_urls:
            chat_assets = _assets_from_urls(user_image_urls, user_message)
        elif _needs_live_data(user_message):
            # Priority 2: live NASA data query — fetch via MCP
            try:
                from agent.data_agent import DataAgent
                nasa_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
                qwen_key = os.environ.get("QWEN_API_KEY", "")
                chat_assets = DataAgent(nasa_key, qwen_key).fetch(user_message)
                retriever = Retriever()
                passages = retriever._extract_passages(chat_assets, top_k=6)
                if passages:
                    grounding = retriever.format_for_prompt(passages)
                    messages.append({"role": "system", "content": grounding})
                else:
                    messages.append({
                        "role": "system",
                        "content": (
                            "A NASA image/data search was performed for this query but "
                            "returned no results. Tell the user that NASA's public databases "
                            "did not return images matching their request. Do not invent "
                            "or describe images you do not have."
                        ),
                    })
            except Exception:
                passages = []
                chat_assets = {}

        # Use simple user/assistant pairs — the messages field stores the full
        # accumulated history per run, so appending all runs' messages lists
        # causes exponential duplication that floods the context window.
        for prev_run in conversation_history[-5:]:
            messages.append({"role": "user", "content": prev_run["user_message"]})
            messages.append({"role": "assistant", "content": prev_run["assistant_response"]})

        messages.append({"role": "user", "content": user_message})

        stream = self.client.stream(messages=messages)
        full_text = ""
        in_think = False

        for chunk in stream:
            if not chunk.choices:
                continue
            text: str = chunk.choices[0].delta.content or ""
            if not text:
                continue
            # Filter <think>…</think> blocks emitted by reasoning-capable models
            while text:
                if in_think:
                    end = text.find("</think>")
                    if end == -1:
                        text = ""  # still inside think block — skip whole chunk
                    else:
                        in_think = False
                        text = text[end + len("</think>"):]
                else:
                    start = text.find("<think>")
                    if start == -1:
                        full_text += text
                        yield text
                        text = ""
                    else:
                        before = text[:start]
                        if before:
                            full_text += before
                            yield before
                        in_think = True
                        text = text[start + len("<think>"):]

        result["answer"] = full_text
        result["retrieved_passages"] = passages
        result["chat_assets"] = chat_assets

    def chat_with_streaming(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ):
        """Yield the full response in one chunk (streaming not yet wired in QwenClient)."""
        result = self.answer(user_message, conversation_history)
        if result["answer"]:
            yield {"text": result["answer"]}
