# Data Source Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AKShare/Tushare data fetching with a unified `shared/data_sources/` registration-pattern layer, adding mootdx (realtime), 腾讯财经 (fundamentals), 同花顺热点 (sector tags) sources.

**Architecture:** DataSource ABC + registry + manager in `shared/data_sources/`. Each source implements `connect/query/health_check/default_rate_limit`. Services call `get_source(name).query(...)` instead of direct API calls. In-memory LRU+TTL cache with configurable rate limits. AKShare preserved as fallback for LHB data only.

**Tech Stack:** Python 3.12, mootdx, httpx, asyncio, Pydantic v2

---

### Task 1: Create shared/data_sources/ base infrastructure

**Files:**
- Create: `shared/data_sources/__init__.py`
- Create: `shared/data_sources/constants.py`
- Create: `shared/data_sources/interface.py`
- Create: `shared/data_sources/registry.py`
- Create: `shared/data_sources/manager.py`

- [ ] **Step 1: Create constants.py**

```python
"""Data source constants — query types, rate limits, cache TTLs."""
from enum import Enum


class QueryType(str, Enum):
    REALTIME_QUOTE = "realtime_quote"
    REALTIME_TICK = "realtime_tick"
    HISTORY_KLINE = "history_kline"
    FUNDAMENTALS = "fundamentals"
    SECTOR_TAGS = "sector_tags"
    HOT_RANKING = "hot_ranking"
    SECTOR_MEMBERS = "sector_members"
    FUND_FLOW = "fund_flow"
    LHB_BOARD = "lhb_board"
    RESEARCH_REPORT = "research_report"
    STOCK_NOTICE = "stock_notice"
    BASIC_INFO = "basic_info"


# Cache TTL in seconds
CACHE_TTL = {
    QueryType.REALTIME_QUOTE: 3,
    QueryType.REALTIME_TICK: 3,
    QueryType.HISTORY_KLINE: 1800,
    QueryType.FUNDAMENTALS: 3600,
    QueryType.SECTOR_TAGS: 300,
    QueryType.HOT_RANKING: 300,
    QueryType.SECTOR_MEMBERS: 300,
    QueryType.FUND_FLOW: 10,
    QueryType.LHB_BOARD: 86400,
    QueryType.RESEARCH_REPORT: 3600,
    QueryType.STOCK_NOTICE: 3600,
    QueryType.BASIC_INFO: 86400,
}

# Rate limits: max calls per second
RATE_LIMITS = {
    "mootdx": 10,
    "tx_finance": 50,
    "ths_hot": 30,
    "eastmoney_notice": 30,
    "eastmoney_report": 30,
}
```

- [ ] **Step 2: Create interface.py**

```python
"""DataSource abstract base class and query/result types."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shared.data_sources.constants import CACHE_TTL, RATE_LIMITS, QueryType


@dataclass
class RateLimit:
    max_per_second: int = 10
    burst: int = 20


@dataclass
class DataQuery:
    query_type: QueryType
    symbols: list[str] = field(default_factory=list)
    date: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataResult:
    source: str
    query_type: QueryType
    data: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    cached_at: float = 0.0
    error: str = ""

    @property
    def is_cached(self) -> bool:
        return self.cached_at > 0

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def count(self) -> int:
        return len(self.data)


class DataSource(ABC):
    """Abstract base for all data sources.

    Subclasses define `name`, `protocol`, and implement the async methods.
    """

    name: str = ""
    protocol: str = "http"

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection. Return True on success."""

    @abstractmethod
    async def query(self, query: DataQuery) -> DataResult:
        """Execute query and return standardized result."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight liveness check."""

    @abstractmethod
    def default_rate_limit(self) -> RateLimit:
        """Return rate limit for this source."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name}) proto={self.protocol}>"
```

- [ ] **Step 3: Create registry.py**

```python
"""DataSource registry — register, lookup, list all sources."""
from __future__ import annotations

from shared.data_sources.interface import DataSource

_registry: dict[str, DataSource] = {}


def register(name: str, source: DataSource) -> None:
    _registry[name] = source


def get(name: str) -> DataSource:
    if name not in _registry:
        raise KeyError(f"DataSource '{name}' not registered. Available: {list(_registry.keys())}")
    return _registry[name]


def list_all() -> list[str]:
    return list(_registry.keys())


def is_registered(name: str) -> bool:
    return name in _registry


def unregister(name: str) -> None:
    _registry.pop(name, None)


def clear() -> None:
    _registry.clear()
```

- [ ] **Step 4: Create manager.py**

```python
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
            return DataResult(source=source_name, query_type=query.query_type,
                              error=f"Source '{source_name}' not registered")

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
                return DataResult(source=source_name, query_type=query.query_type,
                                  error=str(exc))

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


# Singleton
_manager: DataSourceManager | None = None


def get_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
```

- [ ] **Step 5: Create __init__.py**

```python
"""Unified data source layer for dragon-engine.

Usage:
    from shared.data_sources import get_source, DataQuery, QueryType
    source = get_source("mootdx")
    result = await source.query(DataQuery(QueryType.REALTIME_QUOTE, symbols=["000001"]))
"""

from shared.data_sources.constants import CACHE_TTL, RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.data_sources.manager import DataSourceManager, get_manager
from shared.data_sources.registry import clear, get, get as get_source, is_registered, list_all, register, unregister
```

- [ ] **Step 6: Verify base compiles**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -c "from shared.data_sources import DataSource, DataQuery, DataResult, QueryType, register, get_source, list_all, get_manager; print('OK')"
```

---

### Task 2: Implement mootdx data source

**Files:**
- Create: `shared/data_sources/sources/__init__.py`
- Create: `shared/data_sources/sources/mootdx_source.py`

- [ ] **Step 1: Create sources/__init__.py**

```python
"""Data source implementations."""
```

- [ ] **Step 2: Create mootdx_source.py**

```python
"""mootdx data source — TongDaXin TCP realtime quotes via mootdx library."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shared.data_sources.constants import RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.utils.logging import get_logger
from shared.configs.settings import get_settings

logger = get_logger(__name__)

_pool = ThreadPoolExecutor(max_workers=4)


class MootdxSource(DataSource):
    name = "mootdx"
    protocol = "tcp"

    def __init__(self):
        self._client = None
        self._connected = False
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> bool:
        async with self._connect_lock:
            if self._connected:
                return True
            try:
                loop = asyncio.get_running_loop()
                self._client = await loop.run_in_executor(_pool, self._sync_connect)
                self._connected = self._client is not None
                return self._connected
            except Exception as exc:
                logger.warning("mootdx connect failed: %s", exc)
                return False

    @staticmethod
    def _sync_connect():
        try:
            from mootdx.quotes import Quotes
            client = Quotes.factory(market="std", timeout=5)
            client.quotes(symbol=["000001"])  # quick validation
            return client
        except ImportError:
            logger.warning("mootdx not installed — run: pip install mootdx")
            return None
        except Exception as exc:
            logger.warning("mootdx sync connect failed: %s", exc)
            return None

    async def query(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)

        if not self._connected:
            ok = await self.connect()
            if not ok:
                result.error = "mootdx not connected"
                return result

        loop = asyncio.get_running_event_loop()

        if query.query_type in (QueryType.REALTIME_QUOTE, QueryType.REALTIME_TICK):
            return await loop.run_in_executor(_pool, self._fetch_quotes, query)

        if query.query_type == QueryType.HISTORY_KLINE:
            return await loop.run_in_executor(_pool, self._fetch_kline, query)

        if query.query_type == QueryType.FUND_FLOW:
            return await loop.run_in_executor(_pool, self._fetch_fund_flow, query)

        result.error = f"mootdx does not support query_type={query.query_type.value}"
        return result

    def _fetch_quotes(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            market = self._resolve_market(symbols[0])
            client = getattr(self._client, "quotes", None)
            data = self._client.quotes(symbol=symbols) if callable(client) else self._client._quotes(symbols)

            rows = []
            for item in data if isinstance(data, list) else [data]:
                row = {
                    "symbol": str(item.get("code", "")),
                    "name": str(item.get("name", "")),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "price": float(item.get("price", 0)),
                    "volume": float(item.get("volume", 0)),
                    "amount": float(item.get("amount", 0)),
                    "bid1": float(item.get("bid1", 0)),
                    "ask1": float(item.get("ask1", 0)),
                    "time": str(item.get("time", "")),
                }
                rows.append(row)
            result.data = rows
        except Exception as exc:
            result.error = str(exc)
            logger.error("mootdx quote fetch failed: %s", exc)
        return result

    def _fetch_kline(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            market = self._resolve_market(symbols[0])
            rows = []
            for sym in symbols:
                bars = self._client.bars(symbol=sym, frequency=9, offset=100)
                for bar in bars if bars else []:
                    rows.append({
                        "symbol": sym,
                        "date": str(bar.get("date", "")),
                        "open": float(bar.get("open", 0)),
                        "high": float(bar.get("high", 0)),
                        "low": float(bar.get("low", 0)),
                        "close": float(bar.get("close", 0)),
                        "volume": float(bar.get("volume", 0)),
                        "amount": float(bar.get("amount", 0)),
                    })
            result.data = rows
        except Exception as exc:
            result.error = str(exc)
        return result

    def _fetch_fund_flow(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            rows = []
            for sym in symbols:
                data = self._client.quotes(symbol=[sym])
                if data and len(data) > 0:
                    item = data[0]
                    volume = float(item.get("volume", 0))
                    amount = float(item.get("amount", 0))
                    price = float(item.get("price", 0))
                    rows.append({
                        "symbol": str(item.get("code", sym)),
                        "name": str(item.get("name", "")),
                        "price": price,
                        "volume": volume,
                        "amount": amount,
                        "change_pct": float(item.get("change", 0)),
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
            result.data = rows
        except Exception as exc:
            result.error = str(exc)
        return result

    async def health_check(self) -> bool:
        if not self._connected:
            return False
        try:
            loop = asyncio.get_running_event_loop()
            data = await loop.run_in_executor(_pool, self._client.quotes, ["000001"])
            return data is not None and len(data) > 0
        except Exception:
            return False

    @staticmethod
    def _resolve_market(code: str) -> int:
        if code.startswith(("6", "68", "9")):
            return 1  # sh
        return 0  # sz

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("mootdx", 10))
```

- [ ] **Step 3: Verify mootdx source compiles**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -c "from shared.data_sources.sources.mootdx_source import MootdxSource; s = MootdxSource(); print('name:', s.name, 'proto:', s.protocol); print('OK')"
```

---

### Task 3: Implement 腾讯财经 data source

**Files:**
- Create: `shared/data_sources/sources/tx_finance_source.py`

- [ ] **Step 1: Create tx_finance_source.py**

```python
"""腾讯财经 data source — HTTP fundamentals (PE/PB/market cap/ROE) via qt.gtimg.cn."""
from __future__ import annotations

import re

import httpx

from shared.data_sources.constants import RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.utils.logging import get_logger

logger = get_logger(__name__)

TX_BASE = "http://qt.gtimg.cn/q="


class TxFinanceSource(DataSource):
    name = "tx_finance"
    protocol = "http"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10.0)

    async def connect(self) -> bool:
        try:
            resp = await self._client.get(f"{TX_BASE}sh600000")
            return resp.status_code == 200
        except Exception:
            return False

    async def query(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)

        if query.query_type == QueryType.FUNDAMENTALS:
            return await self._fetch_fundamentals(query)

        if query.query_type == QueryType.REALTIME_QUOTE:
            return await self._fetch_light_quotes(query)

        if query.query_type == QueryType.BASIC_INFO:
            return await self._fetch_basic_info(query)

        result.error = f"tx_finance does not support query_type={query.query_type.value}"
        return result

    def _build_codes(self, symbols: list[str]) -> str:
        """Convert ['000001','600000'] → 'sz000001,sh600000'."""
        codes = []
        for s in symbols:
            code = s.strip()
            if code.startswith(("6", "68", "9")):
                codes.append(f"sh{code}")
            else:
                codes.append(f"sz{code}")
        return ",".join(codes)

    async def _fetch_fundamentals(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            codes = self._build_codes(symbols)
            resp = await self._client.get(f"{TX_BASE}{codes}")
            raw = resp.text
            result.raw = raw
            result.data = self._parse_fundamentals(raw)
        except Exception as exc:
            result.error = str(exc)
        return result

    async def _fetch_light_quotes(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            codes = self._build_codes(symbols)
            resp = await self._client.get(f"{TX_BASE}{codes}")
            raw = resp.text
            result.raw = raw
            result.data = self._parse_light_quotes(raw)
        except Exception as exc:
            result.error = str(exc)
        return result

    async def _fetch_basic_info(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            symbols = query.symbols or ["000001"]
            codes = self._build_codes(symbols)
            resp = await self._client.get(f"{TX_BASE}{codes}")
            raw = resp.text
            result.raw = raw
            result.data = self._parse_basic_info(raw)
        except Exception as exc:
            result.error = str(exc)
        return result

    def _parse_fundamentals(self, raw: str) -> list[dict]:
        """Parse qt.gtimg.cn response into standardized fundamental records.

        Response format: var hq_str_sh600000="name,open,close,price,high,low,...,PE,...,PB,...";
        """
        rows = []
        for match in re.finditer(r'hq_str_(\w+)="([^"]*)"', raw):
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) < 50:
                continue
            try:
                symbol = code[2:] if len(code) > 2 else code  # strip sh/sz prefix
                row = {
                    "symbol": symbol,
                    "name": fields[1] if len(fields) > 1 else "",
                    "price": float(fields[3]) if fields[3] else 0,
                    "change_pct": float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                    "pe": float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                    "pb": float(fields[47]) if len(fields) > 47 and fields[47] else 0,
                    "market_cap": float(fields[45]) if len(fields) > 45 and fields[45] else 0,
                    "circulating_cap": float(fields[44]) if len(fields) > 44 and fields[44] else 0,
                    "volume": float(fields[6]) if len(fields) > 6 and fields[6] else 0,
                    "amount": float(fields[37]) if len(fields) > 37 and fields[37] else 0,
                    "high": float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                    "low": float(fields[34]) if len(fields) > 34 and fields[34] else 0,
                    "open": float(fields[5]) if len(fields) > 5 and fields[5] else 0,
                    "pre_close": float(fields[4]) if len(fields) > 4 and fields[4] else 0,
                    "roe": float(fields[42]) if len(fields) > 42 and fields[42] else 0,
                    "eps": float(fields[43]) if len(fields) > 43 and fields[43] else 0,
                    "industry": fields[12] if len(fields) > 12 else "",
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                logger.debug("parse fundamental row failed for %s: %s", code, e)
        return rows

    def _parse_light_quotes(self, raw: str) -> list[dict]:
        rows = []
        for match in re.finditer(r'hq_str_(\w+)="([^"]*)"', raw):
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) < 10:
                continue
            symbol = code[2:] if len(code) > 2 else code
            rows.append({
                "symbol": symbol,
                "name": fields[1],
                "price": float(fields[3]) if fields[3] else 0,
                "change_pct": float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                "volume": float(fields[6]) if len(fields) > 6 and fields[6] else 0,
                "amount": float(fields[37]) if len(fields) > 37 and fields[37] else 0,
            })
        return rows

    def _parse_basic_info(self, raw: str) -> list[dict]:
        rows = []
        for match in re.finditer(r'hq_str_(\w+)="([^"]*)"', raw):
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) < 50:
                continue
            symbol = code[2:] if len(code) > 2 else code
            rows.append({
                "symbol": symbol,
                "name": fields[1],
                "industry": fields[12] if len(fields) > 12 else "",
                "market_cap": float(fields[45]) if len(fields) > 45 and fields[45] else 0,
                "circulating_cap": float(fields[44]) if len(fields) > 44 and fields[44] else 0,
                "total_shares": float(fields[52]) if len(fields) > 52 and fields[52] else 0,
                "list_date": fields[17] if len(fields) > 17 else "",
            })
        return rows

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{TX_BASE}sh600000")
            return resp.status_code == 200 and "hq_str" in resp.text
        except Exception:
            return False

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("tx_finance", 50))

    async def close(self):
        await self._client.aclose()
```

- [ ] **Step 2: Verify tx_finance compiles**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -c "from shared.data_sources.sources.tx_finance_source import TxFinanceSource; s = TxFinanceSource(); print('name:', s.name); print('OK')"
```

---

### Task 4: Implement 同花顺热点 data source

**Files:**
- Create: `shared/data_sources/sources/ths_hot_source.py`

- [ ] **Step 1: Create ths_hot_source.py**

```python
"""同花顺热点 data source — sector/concept tags and hot rankings via HTTP."""
from __future__ import annotations

import json

import httpx

from shared.data_sources.constants import RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# 同花顺 concept board API
THS_CONCEPT_URL = "http://data.10jqka.com.cn/dataapi/block/rank"
THS_STOCK_CONCEPT_URL = "http://d.10jqka.com.cn/v6/concept/index"


class ThsHotSource(DataSource):
    name = "ths_hot"
    protocol = "http"

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://www.10jqka.com.cn/",
            },
        )

    async def connect(self) -> bool:
        try:
            resp = await self._client.get(
                f"{THS_CONCEPT_URL}",
                params={"type": "concept", "page": 1, "size": 10},
            )
            if resp.status_code == 200:
                data = resp.json()
                return "data" in data or "list" in data or "items" in data
            return False
        except Exception:
            return False

    async def query(self, query: DataQuery) -> DataResult:
        result = DataResult(source=self.name, query_type=query.query_type)

        if query.query_type == QueryType.HOT_RANKING:
            return await self._fetch_hot_ranking(query)

        if query.query_type == QueryType.SECTOR_TAGS:
            return await self._fetch_sector_tags(query)

        if query.query_type == QueryType.SECTOR_MEMBERS:
            return await self._fetch_sector_members(query)

        result.error = f"ths_hot does not support query_type={query.query_type.value}"
        return result

    async def _fetch_hot_ranking(self, query: DataQuery) -> DataResult:
        """Fetch top N hot concept boards ranked by heat."""
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            top_n = query.extra.get("top_n", 30)
            # Try multiple page fetches
            rows = []
            page = 1
            while len(rows) < top_n:
                resp = await self._client.get(
                    THS_CONCEPT_URL,
                    params={"type": "concept", "page": page, "size": min(top_n, 50), "order": "desc", "sort": "heat"},
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                items = self._extract_list(data)
                if not items:
                    break
                for item in items:
                    rows.append({
                        "concept_id": str(item.get("id", item.get("code", ""))),
                        "concept_name": str(item.get("name", item.get("concept_name", ""))),
                        "heat": float(item.get("heat", item.get("hot", item.get("hot_score", 0)))),
                        "change_pct": float(item.get("change", item.get("pct", 0))),
                        "stock_count": int(item.get("count", item.get("stock_count", 0))),
                        "leader_stock": str(item.get("leader", item.get("leader_name", ""))),
                        "reason": str(item.get("reason", item.get("desc", ""))),
                    })
                if len(items) < 50:
                    break
                page += 1

            result.data = rows[:top_n]
        except Exception as exc:
            result.error = str(exc)
            logger.error("ths_hot fetch hot ranking failed: %s", exc)
        return result

    async def _fetch_sector_tags(self, query: DataQuery) -> DataResult:
        """Map stock codes to their concept/sector tags.

        Uses 同花顺 individual stock concept page to extract tags.
        """
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            rows = []
            for sym in query.symbols:
                # Try concept mapping page for individual stock
                resp = await self._client.get(
                    f"{THS_STOCK_CONCEPT_URL}",
                    params={"code": sym},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    concepts = self._extract_stock_concepts(data, sym)
                    if concepts:
                        rows.append({
                            "symbol": sym,
                            "concepts": concepts,
                            "concept_count": len(concepts),
                        })
                        continue

                # Fallback: parse from hot ranking to find matching concepts
                hot_result = await self._fetch_hot_ranking(
                    DataQuery(QueryType.HOT_RANKING, extra={"top_n": 50})
                )
                matched = []
                for board in hot_result.data:
                    if sym == board.get("leader_stock", ""):
                        matched.append({
                            "concept_name": board["concept_name"],
                            "heat": board["heat"],
                            "is_leader": True,
                        })
                rows.append({
                    "symbol": sym,
                    "concepts": matched,
                    "concept_count": len(matched),
                })

            result.data = rows
        except Exception as exc:
            result.error = str(exc)
            logger.error("ths_hot sector tags failed: %s", exc)
        return result

    async def _fetch_sector_members(self, query: DataQuery) -> DataResult:
        """Fetch member stocks of a given concept/sector."""
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            concept_id = query.extra.get("concept_id", "")
            concept_name = query.extra.get("concept_name", "")
            resp = await self._client.get(
                f"{THS_STOCK_CONCEPT_URL}",
                params={"code": concept_id or concept_name},
            )
            if resp.status_code == 200:
                data = resp.json()
                members = self._extract_list(data)
                rows = []
                for m in members:
                    rows.append({
                        "symbol": str(m.get("code", m.get("stock_code", ""))),
                        "name": str(m.get("name", m.get("stock_name", ""))),
                        "change_pct": float(m.get("change", 0)),
                    })
                result.data = rows
        except Exception as exc:
            result.error = str(exc)
        return result

    @staticmethod
    def _extract_list(data: dict) -> list:
        """Extract item list from various API response formats."""
        for key in ("data", "list", "items", "result", "content"):
            items = data.get(key)
            if isinstance(items, list):
                return items
            if isinstance(items, dict):
                for sub_key in ("list", "items", "data", "records"):
                    sub = items.get(sub_key)
                    if isinstance(sub, list):
                        return sub
        return []

    @staticmethod
    def _extract_stock_concepts(data: dict, symbol: str) -> list[dict]:
        """Extract concept list for a single stock from API response."""
        items = ThsHotSource._extract_list(data)
        concepts = []
        for item in items:
            concepts.append({
                "concept_name": str(item.get("name", item.get("concept_name", item.get("plate_name", "")))),
                "concept_id": str(item.get("id", item.get("code", item.get("plate_id", "")))),
            })
        return concepts

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                THS_CONCEPT_URL,
                params={"type": "concept", "page": 1, "size": 1},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("ths_hot", 30))

    async def close(self):
        await self._client.aclose()
```

- [ ] **Step 2: Verify ths_hot compiles**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -c "from shared.data_sources.sources.ths_hot_source import ThsHotSource; s = ThsHotSource(); print('name:', s.name); print('OK')"
```

---

### Task 5: Install mootdx and wire up data source auto-registration

**Files:**
- Modify: `shared/configs/settings.py`
- Create: `shared/data_sources/bootstrap.py`

- [ ] **Step 1: Install mootdx**

```bash
D:\K\dragon-engine\venv\Scripts\pip install mootdx -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

- [ ] **Step 2: Add data source config to settings.py**

Read `shared/configs/settings.py`, add these fields to the `Settings` class:

```python
# Data source switches
data_source_realtime: str = "mootdx"      # mootdx | akshare | tx_finance
data_source_fundamentals: str = "tx_finance"  # tx_finance | akshare
data_source_sector: str = "ths_hot"       # ths_hot | none
data_source_lhb: str = "akshare"          # akshare (no alternative yet)
```

- [ ] **Step 3: Create bootstrap.py for auto-registration**

```python
"""Bootstrap data sources — register all available sources at startup."""
from shared.data_sources.registry import register, is_registered
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_sources() -> list[str]:
    """Register all available data sources. Returns list of registered names."""
    registered = []

    # mootdx
    try:
        if not is_registered("mootdx"):
            from shared.data_sources.sources.mootdx_source import MootdxSource
            register("mootdx", MootdxSource())
            registered.append("mootdx")
    except ImportError as e:
        logger.warning("mootdx not available: %s", e)

    # tx_finance
    try:
        if not is_registered("tx_finance"):
            from shared.data_sources.sources.tx_finance_source import TxFinanceSource
            register("tx_finance", TxFinanceSource())
            registered.append("tx_finance")
    except Exception as e:
        logger.warning("tx_finance not available: %s", e)

    # ths_hot
    try:
        if not is_registered("ths_hot"):
            from shared.data_sources.sources.ths_hot_source import ThsHotSource
            register("ths_hot", ThsHotSource())
            registered.append("ths_hot")
    except Exception as e:
        logger.warning("ths_hot not available: %s", e)

    return registered
```

- [ ] **Step 4: Auto-bootstrap in graph_service main.py**

Modify `services/graph_service/main.py` lifespan to call `bootstrap_sources()`:

```python
from shared.data_sources.bootstrap import bootstrap_sources

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Compiling graph ...")
    get_graph()
    logger.info("Graph ready.")
    sources = bootstrap_sources()
    logger.info("Data sources registered: %s", sources)
    yield
```

---

### Task 6: Rewire capital_flow node to use mootdx + tx_finance

**Files:**
- Modify: `services/graph_service/nodes/capital_flow.py`

Read the existing file, then replace the AKShare-based fetching with the new data source layer. Keep the old code path as fallback.

Key changes to the node function:

```python
from shared.data_sources import get_source, DataQuery, QueryType
from shared.data_sources.manager import get_manager

# In the node:
manager = get_manager()

# Replace akshare individual fund flow with mootdx
mootdx = get_source("mootdx")
quote_result = await manager.query("mootdx", DataQuery(
    QueryType.REALTIME_QUOTE, symbols=symbols
))

# Replace akshare market flow — use tx_finance for big picture
tx = get_source("tx_finance")
# For fundamentals context
fund_result = await manager.query("tx_finance", DataQuery(
    QueryType.FUNDAMENTALS, symbols=symbols
))

# Keep akshare as fallback for fund_flow
if not quote_result.data or quote_result.is_error:
    # fallback to original akshare call
    ...
```

- [ ] **Step 1: Read current capital_flow.py**

- [ ] **Step 2: Rewrite with new data sources + akshare fallback**

- [ ] **Step 3: Verify compile**

---

### Task 7: Rewire ingest_event node to add 同花顺 sector tags

**Files:**
- Modify: `services/graph_service/nodes/ingest_event.py`
- Modify: `shared/schemas/agent_state.py` (add sector_tags field)

- [ ] **Step 1: Add sector_tags to AgentState**

Add to the TypedDict:
```python
sector_tags: list[dict]  # 同花顺题材标签
```

- [ ] **Step 2: Add sector tag fetching to ingest_event node**

After the existing event extraction, add:

```python
from shared.data_sources import get_source, DataQuery, QueryType
from shared.data_sources.manager import get_manager

manager = get_manager()
ths = get_source("ths_hot")

# Fetch sector tags for watchlist stocks
sector_result = await manager.query("ths_hot", DataQuery(
    QueryType.SECTOR_TAGS, symbols=watchlist
))

if not sector_result.is_error:
    # Attach sector tags to each event matching the symbol
    for event in events:
        sym = event.get("symbol", "")
        for tag_row in sector_result.data:
            if tag_row["symbol"] == sym:
                event["sector_tags"] = tag_row.get("concepts", [])
```

- [ ] **Step 3: Verify compile**

---

### Task 8: Rewire generate_candidates to add sector tag boost

**Files:**
- Modify: `services/graph_service/nodes/generate_candidates.py`

Add sector_tag_boost to the scoring formulas. Read the current file to find exact line numbers.

Changes:
```python
# After existing leader_score calculation, add:
# Sector tag boost: stocks on current hot concepts get bonus
sector_tag_boost = 0.0
hot_concepts = sector_tags_data.get("hot_concepts", [])
stock_sector_tags = candidate_data.get("sector_tags", [])

for tag in stock_sector_tags:
    for hot in hot_concepts:
        if tag.get("concept_name", "") == hot.get("concept_name", ""):
            sector_tag_boost := max(sector_tag_boost, 0.10)
        if tag.get("is_leader"):
            sector_tag_boost := max(sector_tag_boost, 0.15)

leader_score += sector_tag_boost

# Update reasoning to mention sector tags
if sector_tag_boost > 0:
    reasoning_parts.append(f"蹭上热题材(+{sector_tag_boost:.2f})")
```

- [ ] **Step 1: Read current generate_candidates.py scoring section**

- [ ] **Step 2: Add sector_tag_boost to scoring**

- [ ] **Step 3: Verify compile**

---

### Task 9: End-to-end verification

- [ ] **Step 1: Full compile check**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -m compileall -q . -x venv -x node_modules -x __pycache__ 2>&1; echo "Exit: $?"
```

- [ ] **Step 2: Run demo pipeline**

```bash
cd D:\K\dragon-engine && D:\K\dragon-engine\venv\Scripts\python -m services.graph_service.main 2>&1
```

Expected: All 5 nodes execute, new data sources log connection status, leader candidates produced with sector tags.

- [ ] **Step 3: Verify API server starts**

```bash
cd D:\K\dragon-engine && timeout 10 D:\K\dragon-engine\venv\Scripts\python -c "
import asyncio, sys
sys.path.insert(0, '.')
from services.graph_service.main import app
print('FastAPI app loaded OK')
print('Routes:', [r.path for r in app.routes])
" 2>&1
```
