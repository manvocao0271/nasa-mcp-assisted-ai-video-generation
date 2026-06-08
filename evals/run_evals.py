"""Benchmark suite: tool-selection accuracy, latency, cache hit rate."""

import asyncio
import hashlib
import os
import statistics
import tempfile
import time
import random
from datetime import date, timedelta
from pathlib import Path

from nasa_mcp.features.apod.api import get_apod
from nasa_mcp.cache import Cache
from nasa_mcp.config import Config

from dotenv import load_dotenv


def pick_n_dates(sample_size: int = 30) -> set[date]:
    """Function to pick N random dates since 1995-06-16, returns set of dates."""
    
    start = date(1995, 6, 16)
    end = date.today()
    delta = (end - start).days
    offsets = random.sample(range(delta), k=sample_size)
    return {start + timedelta(days=offset) for offset in offsets}


async def time_call(config: Config, cache: Cache, target_date: date) -> tuple[bool, float]:
    """Returns (was_cache_hit, elapsed_time)."""
    
    key = f"apod:get_apod:{hashlib.sha256(str(target_date).encode()).hexdigest()[:16]}"
    start = time.perf_counter()
    if cache.get(key) is not None:
        end = time.perf_counter()
        return (True, end - start)
    
    response = await get_apod(config, target_date)
    cache.set(key, response, 3600)
    end = time.perf_counter()
    return (False, end - start)


async def main_async() -> None:
    warm, cold = list(), list()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "bench.sqlite3"
        config_temp = Config(os.environ.get("NASA_API_KEY", "DEMO_KEY"), cache_path, 60)
        cache_temp = Cache(cache_path)

        dates = pick_n_dates()
        for day in dates:
            cold.append(await time_call(config_temp, cache_temp, day))
        for day in dates:
            warm.append(await time_call(config_temp, cache_temp, day))
        
        print(f"Hit rate: {cache_temp.stats()['hit_rate']:.2%}")
    
    cold_times = [elapsed for _, elapsed in cold]
    warm_times = [elapsed for _, elapsed in warm]
    
    cold_p50 = statistics.median(cold_times) * 1000 # seconds to ms
    cold_p95 = statistics.quantiles(cold_times, n=20)[18] * 1000

    warm_p50 = statistics.median(warm_times) * 1000
    warm_p95 = statistics.quantiles(warm_times, n=20)[18] * 1000

    print(f"Cold: p50={cold_p50:.1f}ms  p95={cold_p95:.1f}ms")
    print(f"Warm: p50={warm_p50:.3f}ms  p95={warm_p95:.3f}ms")

    speedup = cold_p50 / warm_p50 if warm_p50 > 0 else float("inf")
    print(f"Cache speedup: {speedup:.0f}x")



def main() -> None:
    """Benchmark response time for APOD cold calls and cache hits."""
    load_dotenv()
    print("BENCHMARK")
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
