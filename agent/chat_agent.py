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

        for prev_run in conversation_history[-5:]:
            if prev_run.get("messages"):
                for msg in prev_run["messages"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            else:
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
    def _wants_video(user_message: str, answer: str) -> bool:
        user_lower = (user_message or "").lower()
        answer_lower = (answer or "").lower()

        if any(re.search(p, user_lower) for p in _USER_VIDEO_PATTERNS):
            return True

        return any(phrase in answer_lower for phrase in _ASSISTANT_VIDEO_OFFERS)

    def chat_with_streaming(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ):
        """Yield the full response in one chunk (streaming not yet wired in QwenClient)."""
        result = self.answer(user_message, conversation_history)
        if result["answer"]:
            yield {"text": result["answer"]}
