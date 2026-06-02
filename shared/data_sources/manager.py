"""DataSourceManager — cache, rate limiting, lifecycle for all sources."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from shared.data_sources.constants import CACHE_TTL, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource
from shared.data_sources.registry import _registry, is_registered
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class DataSourceManager:
    """Manages lifecycle of registered data sources with caching."""

    def __init__(self):
        self._cache: dict[str, tuple[float, DataResult]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._rate_trackers: dict[str, list[float]] = {}

    def _cache_key(self, source_name: str, query: DataQuery) -> str:
        symbols_key = ",".join(sorted(query.symbols)) if query.symbols else "all"
        return f"{source_name}:{query.query_type.value}:{query.date}:{symbols_key}"

    def _get_cached(self, key: str, ttl: int) -> DataResult | None:
        if key not in self._cache:
            return None
        ts, result = self._cache[key]
        if time.time() - ts > ttl:
            del self._cache[key]
            return None
        return result

    def _set_cache(self, key: str, result: DataResult) -> None:
        result.cached_at = time.time()
        self._cache[key] = (time.time(), result)
        if len(self._cache) > 20480:
            self._cache.popitem(last=False)

    async def _rate_limit(self, source_name: str) -> None:
        from shared.data_sources.constants import RATE_LIMITS

        max_rps = RATE_LIMITS.get(source_name, 10)
        now = time.time()
        if source_name not in self._rate_trackers:
            self._rate_trackers[source_name] = []
        calls = [t for t in self._rate_trackers[source_name] if now - t < 1.0]
        self._rate_trackers[source_name] = calls
        if len(calls) >= max_rps:
            wait = 1.0 - (now - calls[0])
            if wait > 0:
                await asyncio.sleep(wait)
        self._rate_trackers[source_name].append(time.time())

    async def query(self, source_name: str, query: DataQuery) -> DataResult:
        if not is_registered(source_name):
            return DataResult(
                source=source_name, query_type=query.query_type,
                error=f"Source '{source_name}' not registered"
            )

        ttl = CACHE_TTL.get(query.query_type, 300)
        cache_key = self._cache_key(source_name, query)

        cached = self._get_cached(cache_key, ttl)
        if cached is not None:
            logger.debug("[cache hit] %s", cache_key)
            return cached

        if cache_key not in self._locks:
            self._locks[cache_key] = asyncio.Lock()

        async with self._locks[cache_key]:
            cached = self._get_cached(cache_key, ttl)
            if cached is not None:
                return cached

            await self._rate_limit(source_name)
            source = _registry[source_name]
            try:
                result = await source.query(query)
                self._set_cache(cache_key, result)
                return result
            except Exception as exc:
                logger.error("[%s] query failed: %s", source_name, exc)
                return DataResult(
                    source=source_name, query_type=query.query_type, error=str(exc)
                )

    async def connect_all(self) -> dict[str, bool]:
        results = {}
        for name, source in _registry.items():
            try:
                await source.connect()
                results[name] = True
                logger.info("[%s] connected", name)
            except Exception as exc:
                results[name] = False
                logger.warning("[%s] connect failed: %s", name, exc)
        return results

    async def health_check_all(self) -> dict[str, bool]:
        results = {}
        for name, source in _registry.items():
            try:
                results[name] = await source.health_check()
            except Exception:
                results[name] = False
        return results


_manager: DataSourceManager | None = None


def get_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
