"""Monster stock matcher — candidate-vs-history similarity comparison.

L1: Filter by primary_type / market_cap range / sector
L2: Feature vector similarity (market_cap, turnover, price, etc.)
L3: ChromaDB text embedding similarity on markdown content (sentence-transformers)

Usage:
    matcher = MonsterMatcher(session)
    matches = await matcher.find_similar(
        market_cap=55.0,
        turnover_pct=4.2,
        price=12.5,
        sector="消费电子",
        top_k=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.utils.logging import get_logger
from db.models import MonsterStock

logger = get_logger(__name__)


@dataclass
class CandidateFeatures:
    """Features extracted from a current candidate stock for matching."""
    symbol: str = ""
    stock_name: str = ""
    market_cap: float = 0.0          # 亿
    turnover_pct: float = 0.0        # 最近日均换手率 %
    price: float = 0.0               # 当前价格
    change_pct: float = 0.0          # 最近涨跌幅 %
    sector: str = ""                 # 所属板块
    main_force_net: float = 0.0      # 主力净流入(万元)
    limit_up_count: int = 0          # 近期涨停次数
    is_small_cap: bool = True
    institution_holding_low: bool = True


@dataclass
class MatchResult:
    """A single match between a candidate and a historical monster."""
    monster: MonsterStock
    similarity_score: float          # 0-1 overall similarity
    feature_scores: dict = field(default_factory=dict)  # per-dimension scores
    match_reasons: list = field(default_factory=list)    # human-readable reasons


# Chinese labels for primary/secondary types stored in English
_TYPE_CN: dict[str, str] = {
    "theme_speculation": "题材炒作",
    "earnings_driven": "业绩驱动",
    "policy_driven": "政策驱动",
    "industry_trend": "产业趋势",
    "restructuring": "重组并购",
    "ipo_speculation": "新股投机",
    "sentiment_leader": "情绪龙头",
    "short_term": "短线",
    "long_term": "长线",
    "swing": "波段",
    "small_cap": "小盘",
    "mid_cap": "中盘",
    "limit_up": "连板",
}

def _type_cn(t: str) -> str:
    return _TYPE_CN.get(t, t)


class MonsterMatcher:
    """Multi-layer similarity matcher for monster stock case-based reasoning."""

    # Weights for structured feature similarity
    W_MARKET_CAP = 0.25
    W_TURNOVER = 0.20
    W_PRICE = 0.10
    W_GAIN_PCT = 0.05
    W_SECTOR = 0.15
    W_TYPE = 0.15
    W_TAGS = 0.10

    def __init__(self, session: AsyncSession):
        self._session = session
        self._monsters: list[MonsterStock] = []

    async def _load_monsters(self):
        """Load all monsters from DB (cached per matcher instance)."""
        if self._monsters:
            return
        result = await self._session.execute(select(MonsterStock))
        self._monsters = list(result.scalars().all())

    async def find_similar(
        self,
        features: CandidateFeatures,
        top_k: int = 5,
        filter_types: Optional[list[str]] = None,
        use_l3: bool = True,
    ) -> list[MatchResult]:
        """Find top-k most similar historical monsters.

        Args:
            features: Candidate stock features extracted from pipeline data.
            top_k: Number of top matches to return.
            filter_types: Optional list of primary_types to include.
        """
        await self._load_monsters()

        results: list[MatchResult] = []

        for monster in self._monsters:
            if filter_types and monster.primary_type not in filter_types:
                continue

            scores = {}
            reasons = []

            # 1. Market cap similarity (log scale — difference of orders of magnitude)
            mc1, mc2 = features.market_cap, monster.market_cap_start
            if mc1 > 0 and mc2 > 0:
                mc_ratio = min(mc1, mc2) / max(mc1, mc2) if max(mc1, mc2) > 0 else 0
                scores["market_cap"] = mc_ratio
                if mc_ratio > 0.7:
                    reasons.append(f"市值接近(候选{mc1:.0f}亿 vs {monster.stock_name}{mc2:.0f}亿)")
            else:
                scores["market_cap"] = 0.5

            # 2. Turnover similarity
            t1, t2 = features.turnover_pct, monster.daily_turnover_avg_pre
            if t1 > 0 and t2 > 0:
                t_ratio = min(t1, t2) / max(t1, t2) if max(t1, t2) > 0 else 0
                scores["turnover"] = t_ratio
                if t_ratio > 0.6:
                    reasons.append(f"换手率相似(候选{t1:.1f}% vs {monster.stock_name}{t2:.1f}%)")
            else:
                scores["turnover"] = 0.5

            # 3. Price similarity
            p1, p2 = features.price, float(monster.start_price or 0)
            if p1 > 0 and p2 > 0:
                p_ratio = min(p1, p2) / max(p1, p2) if max(p1, p2) > 0 else 0
                scores["price"] = p_ratio
            else:
                scores["price"] = 0.5

            # 4. Gain potential (monster's historical gain as a ceiling reference)
            # Not similarity, but higher gain monster → more interesting reference
            scores["gain_ref"] = min(float(monster.max_gain_pct or 0) / 1500.0, 1.0)

            # 5. Sector overlap (substring match)
            sector_score = 0.0
            if features.sector and monster.sector:
                s1 = features.sector.lower()
                s2 = monster.sector.lower()
                # Check substring overlap
                if s1 in s2 or s2 in s1:
                    sector_score = 1.0
                else:
                    # Word-level Jaccard
                    w1 = set(s1.replace("/", " ").split())
                    w2 = set(s2.replace("/", " ").split())
                    if w1 and w2:
                        sector_score = len(w1 & w2) / len(w1 | w2)
                scores["sector"] = sector_score
                if sector_score > 0.3:
                    reasons.append(f"板块相近(候选'{features.sector}' vs {monster.stock_name}'{monster.sector}')")
            else:
                scores["sector"] = 0.3

            # 6. Primary type matching (candidate inferred type vs monster type)
            # For now: all types get equal weight; filtered by filter_types above
            scores["type"] = 0.5  # neutral — no candidate type inference yet

            # 7. Tag overlap
            tag_score = 0.0
            if features.sector and monster.tags:
                sector_lower = features.sector.lower()
                matched = sum(1 for t in monster.tags if t.lower() in sector_lower or sector_lower in t.lower())
                if matched > 0:
                    tag_score = min(matched / 3.0, 1.0)
                scores["tags"] = tag_score
            else:
                scores["tags"] = 0.0

            # Weighted total
            total = (
                self.W_MARKET_CAP * scores.get("market_cap", 0) +
                self.W_TURNOVER * scores.get("turnover", 0) +
                self.W_PRICE * scores.get("price", 0) +
                self.W_GAIN_PCT * scores.get("gain_ref", 0) +
                self.W_SECTOR * scores.get("sector", 0) +
                self.W_TYPE * scores.get("type", 0) +
                self.W_TAGS * scores.get("tags", 0)
            )

            # Always provide at least one specific reason (pick best dimension)
            if not reasons:
                best_dim = max(scores, key=scores.get)
                dim_labels = {
                    "market_cap": "市值规模",
                    "turnover": "换手特征",
                    "price": "价格区间",
                    "sector": "板块属性",
                    "type": "驱动类型",
                    "gain_ref": "历史涨幅",
                    "tags": "概念标签",
                }
                reasons.append(dim_labels.get(best_dim, best_dim) + "匹配")

            # Boost: same primary_type
            if filter_types and monster.primary_type in filter_types:
                total *= 1.1

            results.append(MatchResult(
                monster=monster,
                similarity_score=total,
                feature_scores=scores,
                match_reasons=reasons[:3],
            ))

        # ---- L3: ChromaDB text embedding similarity ----
        l3_scores: dict[int, float] = {}  # result index -> text_similarity
        if use_l3:
            try:
                from services.monster_matcher.embedding_store import (
                    build_candidate_query_text,
                    ensure_index,
                    query_similar,
                )
                ensure_index()
                query_text = build_candidate_query_text(features)
                l3_results = query_similar(query_text, top_k=top_k * 3)
                for l3r in l3_results:
                    # Match back to monster by stock_code
                    for mi, mr in enumerate(results):
                        if mr.monster.stock_code == l3r["stock_code"]:
                            l3_scores[mi] = l3r["similarity"]
                            break
            except Exception as e:
                logger.warning("[MonsterMatcher] L3 skipped: %s", e)

        # Fuse L2 + L3: 0.5 * structured + 0.5 * text_similarity
        for mi, mr in enumerate(results):
            text_sim = l3_scores.get(mi, 0.0)
            if text_sim > 0:
                mr.similarity_score = 0.5 * mr.similarity_score + 0.5 * text_sim
                mr.feature_scores["text_semantic"] = text_sim
                if text_sim > 0.6:
                    mr.match_reasons.insert(0, "语义高度相似")

        # Sort by similarity descending, take top_k
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    async def get_context_for_llm(
        self,
        features: CandidateFeatures,
        top_k: int = 3,
    ) -> str:
        """Generate a context string for LLM analogy reasoning.

        Includes full markdown summary + structured comparison.
        """
        matches = await self.find_similar(features, top_k=top_k)
        if not matches:
            return "未找到相似历史妖股案例。"

        lines = ["## 历史妖股案例匹配\n"]
        lines.append(f"候选股: {features.stock_name}({features.symbol}) "
                      f"市值{features.market_cap:.0f}亿 换手{features.turnover_pct:.1f}% "
                      f"板块'{features.sector}'\n")

        for i, mr in enumerate(matches, 1):
            m = mr.monster
            lines.append(f"### 案例{i}: {m.stock_name}({m.stock_code}) "
                         f"相似度{mr.similarity_score:.1%}")
            lines.append(f"- 类型: {m.primary_type}/{m.secondary_type}")
            lines.append(f"- 涨幅: +{m.max_gain_pct}% 在{m.trading_days}个交易日")
            lines.append(f"- 启动市值: {m.market_cap_start:.0f}亿 → 峰值{m.market_cap_peak:.0f}亿")
            lines.append(f"- 启动换手: {m.daily_turnover_avg_pre:.1f}% → 主升换手{m.daily_turnover_avg_surge:.1f}%")
            lines.append(f"- 标签: {', '.join(m.tags[:6])}")
            if mr.match_reasons:
                lines.append(f"- 匹配理由: {'; '.join(mr.match_reasons)}")

            # Include markdown excerpt if available
            md_path = m.markdown_path
            if md_path:
                try:
                    from pathlib import Path
                    full_path = Path(__file__).resolve().parent.parent / md_path
                    if full_path.exists():
                        md_text = full_path.read_text(encoding="utf-8")
                        # Extract just the "关键特征总结" section
                        import re
                        m_section = re.search(
                            r"## 六、关键特征总结.*?(?=---|\Z)",
                            md_text, re.DOTALL
                        )
                        if m_section:
                            excerpt = m_section.group(0)[:800]
                            lines.append(f"\n{excerpt}\n")
                except Exception:
                    pass
            lines.append("")

        return "\n".join(lines)
