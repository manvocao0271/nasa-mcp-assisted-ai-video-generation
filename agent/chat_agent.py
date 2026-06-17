"""Chat agent — multi-turn conversation with optional NASA asset grounding.

Handles user inquiries about the universe with persistent conversation history.
Optionally triggers video generation when the user explicitly asks for a video.
"""

from __future__ import annotations

import os
import re

from agent.qwen_client import QwenClient
from agent.retriever import Retriever


SYSTEM_PROMPT = """You are an expert astronomy guide and universe educator.

When cached NASA data is provided below, you may reference it. You do not call
live NASA APIs during chat — video generation (when requested) fetches fresh data.

When answering:
1. Explain concepts clearly and concisely
2. Use any provided NASA excerpts as grounding when relevant
3. If a visual would help, you may offer: "Would you like me to generate a short video showing this?"

Do not claim you are fetching live NASA data during chat.
"""

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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        passages = []
        try:
            if os.path.exists("output/assets.json"):
                retriever = Retriever()
                passages = retriever.retrieve(user_message, top_k=5, resume=True)
                if passages:
                    retrieved_text = retriever.format_for_prompt(passages)
                    if retrieved_text:
                        messages.append({"role": "system", "content": retrieved_text})
        except Exception:
            passages = []

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
        video_topic = user_message[:200] if should_generate else None

        return {
            "answer": answer,
            "should_generate_video": should_generate,
            "video_topic": video_topic,
            "retrieved_passages": passages,
        }

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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        passages: list = []
        try:
            if os.path.exists("output/assets.json"):
                retriever = Retriever()
                passages = retriever.retrieve(user_message, top_k=5, resume=True)
                if passages:
                    retrieved_text = retriever.format_for_prompt(passages)
                    if retrieved_text:
                        messages.append({"role": "system", "content": retrieved_text})
        except Exception:
            passages = []

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
        result["should_generate_video"] = self._wants_video(user_message, full_text)
        result["video_topic"] = user_message[:200]
        result["retrieved_passages"] = passages

    def chat_with_streaming(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ):
        """Yield the full response in one chunk (streaming not yet wired in QwenClient)."""
        result = self.answer(user_message, conversation_history)
        if result["answer"]:
            yield {"text": result["answer"]}
