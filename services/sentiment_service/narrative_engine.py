"""Narrative engine — assesses whether a unified market narrative is forming.

Adapted from:
  - FinGPT market_sentiment.py source_alignment concept
  - TradingAgents-CN news sentiment aggregation logic
"""

from __future__ import annotations

from shared.schemas.agent_state import SentimentScore
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def assess_narrative_unity(scores: list[SentimentScore]) -> dict:
    """Assess whether the market has formed a unified narrative.

    Returns a dict with:
        is_unified: bool — whether a clear narrative direction exists
        narrative_direction: str — 'bullish', 'bearish', or 'mixed'
        unity_score: float — how consistent the narrative is (0-1)
        propagation_score: float — likelihood of spreading (0-1)
        leading_sectors: list[str] — sectors leading the narrative
        hot_keywords: list[str] — most frequent keywords
    """
    if not scores:
        return {
            "is_unified": False,
            "narrative_direction": "mixed",
            "unity_score": 0.0,
            "propagation_score": 0.0,
            "leading_sectors": [],
            "hot_keywords": [],
        }

    # Compute direction alignment
    bullish = sum(1 for s in scores if s.sentiment_score > 0.15)
    bearish = sum(1 for s in scores if s.sentiment_score < -0.15)
    neutral = len(scores) - bullish - bearish
    total = len(scores)

    dominant_count = max(bullish, bearish)
    unity = dominant_count / total if total > 0 else 0.0

    if unity >= 0.7:
        direction = "bullish" if bullish > bearish else "bearish"
        is_unified = True
    elif unity >= 0.5:
        direction = "bullish" if bullish > bearish else "bearish"
        is_unified = False
    else:
        direction = "mixed"
        is_unified = False

    # Propagation: high hype + high consistency = more likely to spread
    avg_hype = sum(s.hype_score for s in scores) / len(scores) if scores else 0.0
    avg_consistency = sum(s.consistency_score for s in scores) / len(scores) if scores else 0.0
    propagation = avg_hype * 0.6 + avg_consistency * 0.4

    # Collect keywords
    kw_counter: dict[str, int] = {}
    for s in scores:
        for kw in s.keywords:
            kw_counter[kw] = kw_counter.get(kw, 0) + 1
    hot_keywords = sorted(kw_counter, key=kw_counter.get, reverse=True)[:20]

    return {
        "is_unified": is_unified,
        "narrative_direction": direction,
        "unity_score": round(unity, 3),
        "propagation_score": round(min(1.0, propagation), 3),
        "leading_sectors": [],  # filled by hot_topic_detector
        "hot_keywords": hot_keywords,
    }


def assess_monster_stock_potential(
    scores: list[SentimentScore],
    narrative: dict,
) -> dict:
    """Assess whether conditions are ripe for monster stocks (妖股).

    Key indicators:
        - High hype scores across multiple stocks
        - Unified bullish narrative
        - Hot keywords related to speculation (连板, 打板, 龙头)
    """
    if not scores:
        return {"potential": 0.0, "conditions": [], "hot_stocks": []}

    conditions: list[str] = []
    potential = 0.0

    # Condition 1: Unified narrative
    if narrative.get("is_unified"):
        conditions.append("统一叙事已形成")
        potential += 0.3
    else:
        conditions.append("叙事分散")

    # Condition 2: High average hype
    avg_hype = sum(s.hype_score for s in scores) / len(scores)
    if avg_hype > 0.4:
        conditions.append(f"炒作热度高 ({avg_hype:.2f})")
        potential += 0.3
    elif avg_hype > 0.2:
        conditions.append(f"炒作热度中等 ({avg_hype:.2f})")
        potential += 0.15

    # Condition 3: Speculative keywords present
    spec_kw = {"连板", "龙头", "妖股", "打板", "接力", "封板", "涨停板", "情绪高标"}
    all_kw = set()
    for s in scores:
        all_kw.update(s.keywords)
    if all_kw & spec_kw:
        conditions.append("投机关键词活跃")
        potential += 0.2

    # Condition 4: Risk not too high
    avg_risk = sum(s.risk_score for s in scores) / len(scores)
    if avg_risk < 0.3:
        conditions.append("风险可控")
        potential += 0.2
    elif avg_risk > 0.6:
        conditions.append("风险较高")
        potential -= 0.15

    # Top hype stocks
    top_hype = sorted(scores, key=lambda s: s.hype_score, reverse=True)[:5]
    hot_stocks = [s.symbol for s in top_hype if s.hype_score > 0.2]

    return {
        "potential": round(min(1.0, max(0.0, potential)), 3),
        "conditions": conditions,
        "hot_stocks": hot_stocks,
    }
