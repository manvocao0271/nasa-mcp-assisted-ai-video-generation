"""SQLite-backed response cache with per-resource TTLs.

Cache key format: ``{api_name}:{tool_name}:{hash_of_params}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sqlite3
import time


class Cache:
    """TTL-aware SQLite cache for NASA API responses."""

    def __init__(self, path: Path) -> None:
        """Opens connection, creates schema, initializes counters."""

        self.connection = sqlite3.connect(path)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS entries(
                key         TEXT PRIMARY KEY,
                value       BLOB NOT NULL,
                expires_at  REAL
            )
        """)
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.path = path

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key`` if present and unexpired."""

        row = self.connection.execute(
            "SELECT value, expires_at FROM entries WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            self.misses += 1
            return None

        value, expires_at = row
        if expires_at < time.time():
            self.misses += 1
            return None

        self.hits += 1
        return json.loads(value)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store ``value`` under ``key`` with a Time To Live attribute."""

        expires_at = time.time() + ttl_seconds
        self.connection.execute(
            "INSERT OR REPLACE INTO entries (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        self.connection.commit()

    def stats(self) -> dict[str, Any]:
        """Return hit rate, entry count, and on-disk size."""
        entries = self.connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        size_bytes = self.path.stat().st_size
        hit_rate = 0.0
        if self.hits + self.misses > 0:
            hit_rate = self.hits / (self.hits + self.misses)

        return {
            "entries": entries,
            "size_bytes": size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
