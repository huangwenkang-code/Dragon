"""同花顺热点 data source — sector/concept tags and hot rankings via HTTP."""
from __future__ import annotations

import httpx

from shared.data_sources.constants import RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.utils.logging import get_logger

logger = get_logger(__name__)

THS_CONCEPT_URL = "http://data.10jqka.com.cn/dataapi/block/rank"
THS_STOCK_CONCEPT_URL = "http://d.10jqka.com.cn/v6/concept/index"


class ThsHotSource(DataSource):
    name = "ths_hot"
    protocol = "http"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "http://www.10jqka.com.cn/",
                },
            )
        return self._client

    async def connect(self) -> bool:
        try:
            client = self._ensure_client()
            resp = await client.get(
                THS_CONCEPT_URL,
                params={"type": "concept", "page": 1, "size": 10},
            )
            if resp.status_code == 200:
                data = resp.json()
                return any(k in data for k in ("data", "list", "items"))
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
            client = self._ensure_client()
            rows = []
            page = 1
            while len(rows) < top_n:
                resp = await client.get(
                    THS_CONCEPT_URL,
                    params={
                        "type": "concept",
                        "page": page,
                        "size": min(top_n, 50),
                        "order": "desc",
                        "sort": "heat",
                    },
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
        """Map stock codes to their concept/sector tags."""
        result = DataResult(source=self.name, query_type=query.query_type)
        try:
            client = self._ensure_client()
            rows = []
            for sym in query.symbols:
                resp = await client.get(
                    THS_STOCK_CONCEPT_URL,
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
            client = self._ensure_client()
            concept_id = query.extra.get("concept_id", "")
            concept_name = query.extra.get("concept_name", "")
            resp = await client.get(
                THS_STOCK_CONCEPT_URL,
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
        items = ThsHotSource._extract_list(data)
        concepts = []
        for item in items:
            concepts.append({
                "concept_name": str(item.get(
                    "name", item.get("concept_name", item.get("plate_name", ""))
                )),
                "concept_id": str(item.get(
                    "id", item.get("code", item.get("plate_id", ""))
                )),
            })
        return concepts

    async def health_check(self) -> bool:
        try:
            client = self._ensure_client()
            resp = await client.get(
                THS_CONCEPT_URL,
                params={"type": "concept", "page": 1, "size": 1},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def default_rate_limit(self) -> RateLimit:
        return RateLimit(max_per_second=RATE_LIMITS.get("ths_hot", 30))

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
