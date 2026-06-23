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
    return f"""You are an expert astronomy guide and universe educator backed by live NASA data.
Today's date is {today}.

When NASA data is provided below in a "Retrieved NASA data" block, treat it as ground truth
and cite specific facts, titles, dates, or descriptions from it in your answer.

When answering:
1. Explain concepts clearly and concisely
2. Use any provided NASA data as grounding — prefer it over your training knowledge for current events
3. If a visual would help, you may offer: "Would you like me to generate a short video showing this?"

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


# User must explicitly ask for video — not broad phrases like "show me" alone.
_USER_VIDEO_PATTERNS = (
    r"\bgenerate\s+(a\s+)?video\b",
    r"\bmake\s+(a\s+)?video\b",
    r"\bcreate\s+(a\s+)?video\b",
    r"\bproduce\s+(a\s+)?video\b",
    r"\bshort\s+video\b",
    r"\bshort\s+film\b",
    r"\bvideo\s+(about|of|showing|on)\b",
    r"\bfilm\s+(about|of|showing|on)\b",
    r"\banimate\b",
    r"\bwhat\s+it'?s?\s+like\s+(on|at|in)\b.+\b(video|film|clip)\b",
    r"\bshow\s+me.+\b(video|film|clip)\b",
)

_ASSISTANT_VIDEO_OFFERS = (
    "would you like me to generate a short video",
    "would you like me to generate a video",
    "i can generate a short video",
    "i can generate a video showing",
    "generate a short video showing",
)

# Short affirmative replies that confirm the assistant's video offer.
_AFFIRMATIVE_PATTERNS = (
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do\s+it|absolutely|definitely|of\s+course|sounds\s+good|let'?s?\s+do\s+it|go\s+for\s+it)\s*[.!]?\s*$",
    r"^\s*yes[,\s]+please\s*[.!]?\s*$",
    r"^\s*please\s+(do|generate|make|create)\s+(it|that|a\s+video)\s*[.!]?\s*$",
)


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

        should_generate = self._wants_video(user_message, answer)
        if should_generate:
            video_topic: str | None = self._extract_specific_topic(answer, user_message)
        else:
            video_topic = None

        return {
            "answer": answer,
            "should_generate_video": should_generate,
            "video_topic": video_topic,
            "retrieved_passages": passages,
            "chat_assets": chat_assets,
        }

    def _extract_specific_topic(self, answer: str, fallback: str) -> str:
        """Use an LLM call to distil a precise 5-10 word NASA search query from *answer*.

        E.g. if the answer discusses WASP-76b in detail, returns
        "WASP-76b ultra-hot exoplanet" rather than "exoplanet".
        Falls back to *fallback* on any error.
        """
        try:
            extraction_msgs = [
                {
                    "role": "system",
                    "content": (
                        "Extract the single main astronomical subject from the text below as a short "
                        "NASA image search query (5–10 words max). "
                        "Include the specific object name, spacecraft/telescope, or phenomenon. "
                        "Reply with ONLY the search query — no explanation, no punctuation around it."
                    ),
                },
                {"role": "user", "content": answer[:800]},
            ]
            resp = self.client.chat(messages=extraction_msgs)
            topic = (resp.choices[0].message.content or "").strip().strip('"\'')
            return topic[:200] if topic else fallback[:200]
        except Exception:
            return fallback[:200]

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        """Return True if *text* is a short affirmative reply."""
        return any(re.search(p, text.strip(), re.IGNORECASE) for p in _AFFIRMATIVE_PATTERNS)

    @staticmethod
    def _wants_video(user_message: str, answer: str) -> bool:
        user_lower = (user_message or "").lower()
        answer_lower = (answer or "").lower()

        if any(re.search(p, user_lower) for p in _USER_VIDEO_PATTERNS):
            return True

        return any(phrase in answer_lower for phrase in _ASSISTANT_VIDEO_OFFERS)

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
        - ``should_generate_video`` – bool
        - ``video_topic``          – str
        - ``retrieved_passages``   – list[dict]

        Thinking tokens enclosed in ``<think>…</think>`` are filtered out so
        they never reach the UI, even when a reasoning-capable model is used.
        """
        if conversation_history is None:
            conversation_history = []

        # ── Affirmative shortcut ──────────────────────────────────────────────
        # If the user is just saying "yes/sure/please" in response to the
        # assistant's previous video offer, skip the LLM call entirely and
        # trigger video generation immediately.
        last_run = conversation_history[-1] if conversation_history else None
        if last_run and self._is_affirmative(user_message):
            last_assistant = last_run.get("assistant_response", "")
            if any(phrase in last_assistant.lower() for phrase in _ASSISTANT_VIDEO_OFFERS):
                topic = last_run.get("user_message", user_message)[:200]
                msg = "Let's generate that video! Starting now…"
                result["answer"] = msg
                result["should_generate_video"] = True
                result["video_topic"] = topic
                result["retrieved_passages"] = []
                yield msg
                return

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
        should_generate = self._wants_video(user_message, full_text)
        result["should_generate_video"] = should_generate
        # Extract a specific, targeted NASA search query from the answer so the
        # DataAgent fetches images for the right subject (e.g. "WASP-76b ultra-hot
        # Jupiter" instead of the vague user message "show me an interesting exoplanet").
        if should_generate:
            video_topic = self._extract_specific_topic(full_text, user_message)
        else:
            video_topic = user_message[:200]
        result["video_topic"] = video_topic
        result["retrieved_passages"] = passages
        result["chat_assets"] = chat_assets
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
