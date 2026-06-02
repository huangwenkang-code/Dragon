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
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def connect(self) -> bool:
        try:
            client = self._ensure_client()
            resp = await client.get(f"{TX_BASE}sh600000")
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
            client = self._ensure_client()
            resp = await client.get(f"{TX_BASE}{codes}")
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
            client = self._ensure_client()
            resp = await client.get(f"{TX_BASE}{codes}")
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
            client = self._ensure_client()
            resp = await client.get(f"{TX_BASE}{codes}")
            raw = resp.text
            result.raw = raw
            result.data = self._parse_basic_info(raw)
        except Exception as exc:
            result.error = str(exc)
        return result

    def _parse_fundamentals(self, raw: str) -> list[dict]:
        rows = []
        for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) < 50:
                continue
            try:
                symbol = code[2:] if len(code) > 2 else code
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
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                logger.debug("parse fundamental row failed for %s: %s", code, e)
        return rows

    def _parse_light_quotes(self, raw: str) -> list[dict]:
        rows = []
        for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
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
        for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
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
            client = self._ensure_client()
            resp = await client.get(f"{TX_BASE}sh600000")
            return resp.status_code == 200 and "hq_str" in resp.text
        except Exception:
            return False

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("tx_finance", 50))

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
