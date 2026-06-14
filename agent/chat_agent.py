"""Chat agent — multi-turn conversation with NASA tool context.

Handles user inquiries about the universe with persistent conversation history.
Maintains context across turns and optionally triggers video generation for complex queries.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from agent.qwen_client import QwenClient
from agent.retriever import Retriever


SYSTEM_PROMPT = """You are an expert astronomy guide and universe educator.
You have access to NASA tools including:
- APOD (Astronomy Picture of the Day)
- DONKI (space weather events)
- EPIC (Earth imagery)
- NEO (near-Earth objects)
- Exoplanet database
- NASA Image Library

When answering questions:
1. Use your knowledge to explain concepts clearly
2. Reference NASA data when relevant
3. Suggest visual examples (say "I can generate a video showing..." when appropriate)
4. Keep responses concise but informative

For complex topics or when the user wants visual learning, suggest: "Would you like me to generate a short video showing this?"
"""


class ChatAgent:
    """Maintains multi-turn conversations with astronomy context."""

    def __init__(self, qwen_client: QwenClient):
        self.client = qwen_client
        self.model = os.environ.get("QWEN_CHAT_MODEL", "qwen3.7-plus")

    def answer(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Answer a user query with conversation context.

        Args:
            user_message: The user's current question
            conversation_history: Previous turns (optional)

        Returns:
            {
                "answer": str,  # Assistant response
                "should_generate_video": bool,  # If user wants a video
                "video_topic": str | None,  # What to make a video about
            }
        """
        if conversation_history is None:
            conversation_history = []

        # Rebuild message history for context
        messages = []

        # Add system prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Phase 1 RAG: fetch short grounding passages from NASA assets and
        # inject them as an additional system message to ground the model.
        passages = []
        try:
            retriever = Retriever()
            passages = retriever.retrieve(user_message, top_k=5)
            if passages:
                retrieved_text = retriever.format_for_prompt(passages)
                if retrieved_text:
                    messages.append({"role": "system", "content": retrieved_text})
        except Exception:
            # Swallow retrieval errors — chat should still work without RAG.
            passages = []

        # Add conversation history (limit to last 5 turns to control context)
        for prev_run in conversation_history[-5:]:
            if "messages" in prev_run and prev_run["messages"]:
                for msg in prev_run["messages"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                # Fallback: use user_message and assistant_response
                messages.append({"role": "user", "content": prev_run["user_message"]})
                messages.append({"role": "assistant", "content": prev_run["assistant_response"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Get response from Qwen
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        # Simple heuristic: detect if user wants a video
        should_generate = any(
            phrase in user_message.lower()
            for phrase in ["show me", "generate", "create", "video", "visual", "see a"]
        )

        video_topic = None
        if should_generate:
            # Try to extract what they want to see (very simple)
            video_topic = self._extract_topic(user_message)

        return {
            "answer": answer,
            "should_generate_video": should_generate,
            "video_topic": video_topic,
            "retrieved_passages": passages,
        }

    def _extract_topic(self, user_message: str) -> str:
        """Simple topic extraction from user message."""
        # Just use the message itself as the topic for now
        # In production, this could use NER or another model
        return user_message[:100]

    def chat_with_streaming(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ):
        """
        Stream a chat response token by token.

        Yields dicts with "text" key containing response chunks.
        """
        if conversation_history is None:
            conversation_history = []

        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Phase 1 RAG (streaming): fetch grounding passages synchronously
        try:
            retriever = Retriever()
            passages = retriever.retrieve(user_message, top_k=5)
            if passages:
                retrieved_text = retriever.format_for_prompt(passages)
                if retrieved_text:
                    messages.append({"role": "system", "content": retrieved_text})
        except Exception:
            pass

        for prev_run in conversation_history[-5:]:
            if "messages" in prev_run and prev_run["messages"]:
                for msg in prev_run["messages"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                messages.append({"role": "user", "content": prev_run["user_message"]})
                messages.append({"role": "assistant", "content": prev_run["assistant_response"]})

        messages.append({"role": "user", "content": user_message})

        # Stream from Qwen (if streaming is supported in the client)
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield {"text": chunk.choices[0].delta.content}
