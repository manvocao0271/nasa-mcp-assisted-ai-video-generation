"""Phase-1 Retriever: simple RAG using the existing DataAgent via Orchestrator.

This lightweight retriever fetches `output/assets.json` for a given query by
calling the pipeline's `Orchestrator.fetch_data()` generator and extracts a
small set of short passages (image captions and short tool snippets) that
can be injected into an LLM prompt as grounding evidence.

Phase 1 is intentionally dependency-light: it reuses existing agents and
does not require embeddings or an external vector store. Later phases will
replace or augment this with an embeddings-backed nearest-neighbour search.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from agent.orchestrator import Orchestrator


class Retriever:
    """Lightweight retriever that returns top-K passages from NASA assets.

    Each passage is a dict: {"snippet", "source", "doc_id", "type"}.
    """

    def __init__(self, qwen_api_key: str | None = None, nasa_api_key: str | None = None):
        self.qwen_api_key = qwen_api_key or os.environ.get("QWEN_API_KEY")
        self.nasa_api_key = nasa_api_key or os.environ.get("NASA_API_KEY")

    def retrieve(self, user_message: str, top_k: int = 5, resume: bool = False) -> List[Dict]:
        """Fetch assets for `user_message` and return up to `top_k` passages.

        This calls `Orchestrator.fetch_data` synchronously and extracts short
        snippets from `assets["images"]` and `assets["data"]`.
        """
        if not self.nasa_api_key or not self.qwen_api_key:
            return []

        orchestrator = Orchestrator(qwen_api_key=self.qwen_api_key, nasa_api_key=self.nasa_api_key)

        assets: dict = {}
        try:
            for update in orchestrator.fetch_data(user_message, resume=resume):
                assets = update.get("assets", assets)
                if update.get("stage") == "data" and update.get("status") == "done":
                    assets = update.get("assets", assets)
                    break
        except Exception:
            # Retrieval must not raise — fall back to empty list on errors.
            return []

        passages: List[Dict] = []

        # Image captions are high-value grounding passages for visual queries.
        for img in assets.get("images", []):
            passages.append(
                {
                    "snippet": (img.get("caption") or "").strip(),
                    "source": img.get("source", "image"),
                    "doc_id": img.get("url", ""),
                    "type": "image",
                }
            )

        # Tool outputs: grab brief excerpts (title/explanation) or truncated JSON.
        for tool, result in assets.get("data", {}).items():
            snippet = ""
            try:
                if isinstance(result, dict):
                    snippet = (
                        result.get("title")
                        or result.get("explanation")
                        or json.dumps(result)[:300]
                    )
                elif isinstance(result, list):
                    first = result[0] if result else {}
                    if isinstance(first, dict):
                        snippet = (
                            first.get("title")
                            or first.get("explanation")
                            or json.dumps(first)[:300]
                        )
                    else:
                        snippet = str(first)[:300]
                else:
                    snippet = str(result)[:300]
            except Exception:
                snippet = str(result)[:200]

            passages.append(
                {"snippet": snippet.strip(), "source": tool, "doc_id": f"tool:{tool}", "type": "data"}
            )

        # Deduplicate by doc_id/snippet while preserving order, then trim to top_k.
        seen = set()
        unique: List[Dict] = []
        for p in passages:
            key = (p.get("doc_id") or p.get("snippet"))
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
            if len(unique) >= top_k:
                break

        return unique

    def _extract_passages(self, assets: dict, top_k: int = 5) -> List[Dict]:
        """Extract passages from an already-fetched assets dict (no network call).

        Used by ChatAgent to ground answers from a DataAgent.fetch() result
        without triggering a fresh Orchestrator run or writing to disk.
        """
        passages: List[Dict] = []

        for img in assets.get("images", []):
            caption = (img.get("caption") or "").strip()
            if caption:
                passages.append({
                    "snippet": caption,
                    "source": img.get("source", "image"),
                    "doc_id": img.get("url", ""),
                    "type": "image",
                })

        for tool, result in assets.get("data", {}).items():
            try:
                if isinstance(result, dict):
                    snippet = (
                        result.get("title")
                        or result.get("explanation")
                        or json.dumps(result)[:400]
                    )
                elif isinstance(result, list):
                    first = result[0] if result else {}
                    snippet = (
                        (first.get("title") or first.get("explanation") or json.dumps(first)[:400])
                        if isinstance(first, dict) else str(first)[:300]
                    )
                else:
                    snippet = str(result)[:300]
            except Exception:
                snippet = ""

            if snippet:
                passages.append({
                    "snippet": snippet.strip(),
                    "source": tool,
                    "doc_id": f"tool:{tool}",
                    "type": "data",
                })

        seen: set = set()
        unique: List[Dict] = []
        for p in passages:
            key = p.get("doc_id") or p.get("snippet")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(p)
            if len(unique) >= top_k:
                break

        return unique

    def format_for_prompt(self, passages: List[Dict]) -> str:
        """Return a short human-readable block suitable for injection into prompts."""
        if not passages:
            return ""
        lines = []
        for i, p in enumerate(passages):
            snippet = (p.get("snippet") or "").replace("\n", " ")
            source = p.get("source") or ""
            doc = p.get("doc_id") or ""
            lines.append(f"[{i+1}] {source}: {snippet} {('(' + doc + ')') if doc else ''}")
        return "Retrieved NASA data:\n" + "\n".join(lines)
