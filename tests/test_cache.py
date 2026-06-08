"""Integration tests: cache behavior across re-instantiations and TTL expiry."""

import time
from pathlib import Path

from nasa_mcp.cache import Cache


def test_cache_persists_across_instances(tmp_cache_path: Path) -> None:
    """Writing with one Cache and reading with another returns the same value."""
    first = Cache(tmp_cache_path)
    first.set("apod:get_apod:abc", {"title": "Test"}, ttl_seconds=3600)

    second = Cache(tmp_cache_path)
    assert second.get("apod:get_apod:abc") == {"title": "Test"}


def test_expired_entries_are_treated_as_misses(cache: Cache) -> None:
    """An entry whose expires_at has passed must not be returned."""
    cache.set("apod:get_apod:xyz", {"title": "Stale"}, ttl_seconds=0)
    time.sleep(0.01)

    assert cache.get("apod:get_apod:xyz") is None


def test_stats_reflect_hit_and_miss_counts(cache: Cache) -> None:
    """The stats dict tracks hits and misses across calls in the same process."""
    cache.set("k", {"v": 1}, ttl_seconds=3600)
    cache.get("k")
    cache.get("missing")
    cache.get("k")

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["entries"] >= 1
