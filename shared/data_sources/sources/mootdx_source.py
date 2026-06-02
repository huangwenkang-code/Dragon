"""mootdx data source — TongDaXin TCP realtime quotes via mootdx library."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shared.data_sources.constants import RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.utils.logging import get_logger

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
            client.quotes(symbol=["000001"])
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

        loop = asyncio.get_running_loop()

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
            data = self._client.quotes(symbol=symbols)

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
                    rows.append({
                        "symbol": str(item.get("code", sym)),
                        "name": str(item.get("name", "")),
                        "price": float(item.get("price", 0)),
                        "volume": float(item.get("volume", 0)),
                        "amount": float(item.get("amount", 0)),
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
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(_pool, self._client.quotes, ["000001"])
            return data is not None and len(data) > 0
        except Exception:
            return False

    @staticmethod
    def _resolve_market(code: str) -> int:
        if code.startswith(("6", "68", "9")):
            return 1
        return 0

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("mootdx", 10))
