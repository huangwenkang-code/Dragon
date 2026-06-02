"""Sentiment analyzer — keyword-based + LLM sentiment analysis for A-share news.

Adapted from:
  - FinGPT FinGPT_Sentiment_Analysis_v3 keyword approach
  - FinGPT Forecaster market_sentiment.py aggregation logic
  - TradingAgents-CN AKShareProvider sentiment scoring
"""

from __future__ import annotations

from shared.schemas.agent_state import AgentState, SentimentScore, Event
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Positive/negative keyword dictionaries — adapted from FinGPT & TradingAgents-CN
POSITIVE_KEYWORDS = {
    "涨停": 1.0, "暴涨": 0.9, "大涨": 0.8, "飙升": 0.8, "涨停板": 1.0,
    "创新高": 0.7, "突破": 0.6, "上涨": 0.5, "增长": 0.4, "利好": 0.6,
    "看好": 0.5, "推荐": 0.5, "买入": 0.6, "增持": 0.5, "超预期": 0.7,
    "业绩增长": 0.7, "营收增长": 0.6, "净利润增长": 0.7, "扭亏为盈": 0.8,
    "获批": 0.6, "中标": 0.7, "签约": 0.6, "合作": 0.5, "并购": 0.5,
    "重组": 0.5, "分红": 0.5, "回购": 0.6, "中标": 0.7, "签约": 0.6,
    "强势": 0.5, "龙头": 0.6, "妖股": 0.7, "连板": 0.8, "打板": 0.6,
    "高开": 0.5, "拉升": 0.6, "封板": 0.7, "接力": 0.6,
}

NEGATIVE_KEYWORDS = {
    "跌停": -1.0, "暴跌": -0.9, "大跌": -0.8, "跳水": -0.8, "跌停板": -1.0,
    "创新低": -0.7, "破位": -0.6, "下跌": -0.5, "下滑": -0.4, "利空": -0.6,
    "看空": -0.5, "卖出": -0.6, "减持": -0.5, "警告": -0.5, "低于预期": -0.7,
    "业绩下滑": -0.7, "营收下降": -0.6, "净利润下降": -0.7, "亏损": -0.6,
    "被查": -0.8, "违规": -0.7, "处罚": -0.7, "诉讼": -0.6, "退市": -1.0,
    "停牌": -0.5, "商誉减值": -0.7, "暴雷": -0.9, "炸板": -0.8,
    "弱势": -0.5, "淘汰": -0.4, "退潮": -0.6, "高开低走": -0.7,
}

# Keywords for hype/monster-stock potential
HYPE_KEYWORDS = [
    "连板", "妖股", "龙头", "涨停", "打板", "接力", "封板", "地天板",
    "辨识度", "情绪高标", "抱团", "炒作", "题材", "风口", "热点",
    "强势", "暴涨", "飙升", "连续涨停",
]

RISK_KEYWORDS = [
    "炸板", "退潮", "高开低走", "跌停", "暴雷", "减持", "违规",
    "处罚", "退市", "停牌", "利空", "跳水", "暴跌", "崩盘",
]


def _keyword_sentiment(text: str) -> float:
    """Calculate sentiment score from keyword matches."""
    score = 0.0
    for kw, weight in POSITIVE_KEYWORDS.items():
        if kw in text:
            score += weight
    for kw, weight in NEGATIVE_KEYWORDS.items():
        if kw in text:
            score += weight
    # Normalize to [-1, 1]
    return max(-1.0, min(1.0, score / 3.0))


def _hype_score(text: str) -> float:
    """Calculate hype/monster-stock potential from keywords."""
    matches = sum(1 for kw in HYPE_KEYWORDS if kw in text)
    return min(1.0, matches / 5.0)


def _risk_score(text: str) -> float:
    """Calculate risk score from keywords."""
    matches = sum(1 for kw in RISK_KEYWORDS if kw in text)
    return min(1.0, matches / 4.0)


def _consistency_score(news_items: list[dict]) -> float:
    """Calculate sentiment consistency across multiple news items.

    High consistency = most news agree on direction (positive or negative).
    Low consistency = mixed signals.
    """
    if not news_items:
        return 0.5
    scores = [_keyword_sentiment(f"{i.get('title','')} {i.get('content','')[:500]}") for i in news_items]
    positives = sum(1 for s in scores if s > 0.1)
    negatives = sum(1 for s in scores if s < -0.1)
    neutrals = len(scores) - positives - negatives
    if len(scores) <= 1:
        return 0.5
    dominant = max(positives, negatives) / len(scores)
    return round(dominant, 3)


def analyze_single(
    symbol: str,
    news_items: list[dict],
) -> SentimentScore:
    """Analyze sentiment for a single stock based on its news.

    Uses FinBERT for base sentiment (with keyword fallback).
    Hype/risk/consistency still use keyword heuristics (FinBERT doesn't provide these).
    """
    text = " ".join(
        f"{i.get('title', '')} {i.get('content', '')[:300]}"
        for i in news_items[:10]
    )

    # --- FinBERT base sentiment (primary) ---
    finbert_pos = finbert_neg = finbert_neu = 0.0
    finbert_used = False
    price_penalty = 0.0
    try:
        from services.sentiment_service.finbert_inference import predict_sentiment
        finbert_result = predict_sentiment(text)
        if finbert_result is not None:
            finbert_pos = finbert_result["positive"]
            finbert_neg = finbert_result["negative"]
            finbert_neu = finbert_result["neutral"]

            # Apply price-based calibration: a stock that crashed 50% is not
            # "99% positive" just because the news text says it bounced today.
            try:
                from services.sentiment_service.price_context import get_price_penalty
                price_penalty = get_price_penalty(symbol)
            except Exception:
                price_penalty = 0.0

            finbert_pos_adj = finbert_pos * (1.0 - price_penalty)
            sentiment = round(finbert_pos_adj - finbert_neg, 3)
            finbert_used = True
        else:
            sentiment = _keyword_sentiment(text)
    except Exception:
        sentiment = _keyword_sentiment(text)

    # Merge keyword risk with price-based distress
    hype = _hype_score(text)
    risk = max(_risk_score(text), price_penalty)
    consistency = _consistency_score(news_items)
    narrative_strength = (abs(sentiment) * 0.5 + hype * 0.3 + consistency * 0.2)
    confidence = min(1.0, len(news_items) / 10.0)

    # Extract keywords from text
    all_kw = set()
    for kw in list(POSITIVE_KEYWORDS.keys()) + list(NEGATIVE_KEYWORDS.keys()) + HYPE_KEYWORDS:
        if kw in text:
            all_kw.add(kw)

    logger.debug("[sentiment] %s finbert=%s penalty=%.2f score=%.3f (pos=%.2f→%.2f neg=%.2f neu=%.2f)",
                 symbol, finbert_used, price_penalty, sentiment,
                 finbert_pos, finbert_pos * (1.0 - price_penalty), finbert_neg, finbert_neu)

    return SentimentScore(
        target_id=symbol,
        target_type="stock",
        symbol=symbol,
        sentiment_score=round(sentiment, 3),
        narrative_score=round(narrative_strength, 3),
        hype_score=round(hype, 3),
        consistency_score=round(consistency, 3),
        risk_score=round(risk, 3),
        confidence=round(confidence, 3),
        heat=round((abs(sentiment) * 0.6 + hype * 0.4), 3),
        consensus=round(consistency, 3),
        diffusion_speed=min(1.0, len(news_items) / 20.0),
        narrative_strength=round(narrative_strength, 3),
        keywords=list(all_kw)[:15],
        finbert_positive=round(finbert_pos, 4),
        finbert_negative=round(finbert_neg, 4),
        finbert_neutral=round(finbert_neu, 4),
    )


async def analyze_news_batch(
    events: list[dict],
    news_items: list[dict],
    llm=None,
) -> list[SentimentScore]:
    """Analyze sentiment for all events in batch.

    Uses keyword analysis by default. If llm is provided, also enriches
    the sentiment summary with LLM reasoning.
    """
    scores: list[SentimentScore] = []

    # Group news by symbol
    symbol_news: dict[str, list[dict]] = {}
    for item in news_items:
        for sym in item.get("symbols", []):
            symbol_news.setdefault(sym, []).append(item)

    # For each event, compute sentiment from its symbols' news
    for evt in events:
        symbols = evt.get("symbol_list", []) or [evt.get("event_id", "")]
        event_text = f"{evt.get('title', '')} {evt.get('content', '') or evt.get('summary', '')}"

        for sym in symbols:
            sym_news = symbol_news.get(sym, [])
            if not sym_news:
                # Fallback: use event text directly
                score = SentimentScore(
                    target_id=evt.get("event_id", sym),
                    target_type="event",
                    symbol=sym,
                    sentiment_score=round(_keyword_sentiment(event_text), 3),
                    narrative_score=0.0,
                    hype_score=_hype_score(event_text),
                    consistency_score=0.5,
                    risk_score=_risk_score(event_text),
                    confidence=0.3,
                    heat=0.0,
                    consensus=0.5,
                    diffusion_speed=0.0,
                    narrative_strength=0.0,
                    keywords=evt.get("keywords", []),
                    finbert_positive=0.0,
                    finbert_negative=0.0,
                    finbert_neutral=0.0,
                )
                scores.append(score)
                continue

            score = analyze_single(sym, sym_news)
            score.target_id = evt.get("event_id", sym)
            scores.append(score)

    # Enrich with LLM summary if available
    if llm and scores:
        try:
            top_scores = sorted(scores, key=lambda s: abs(s.sentiment_score), reverse=True)[:5]
            summary_text = "\n".join(
                f"{s.symbol}: 情绪={s.sentiment_score:+.3f} 炒作={s.hype_score:.3f} 风险={s.risk_score:.3f}"
                for s in top_scores
            )
            enrich_prompt = (
                "根据以下A股情绪分析数据，给出一句话总结当前市场情绪和炒作方向：\n"
                f"{summary_text}\n\n一句话总结："
            )
            enriched_model = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""

            response = await llm.ainvoke(enrich_prompt)
            summary = response.content if hasattr(response, "content") else str(response)

            # Store audit trail on enriched scores
            for s in top_scores:
                s.llm_prompt = enrich_prompt
                s.llm_response = summary

            logger.info("LLM sentiment summary: %s", summary[:200])
        except Exception as exc:
            logger.warning("LLM enrichment failed: %s", exc)

    logger.info("analyze_news_batch: %d events → %d scores", len(events), len(scores))
    return scores


async def analyze_sentiment(state: AgentState) -> dict:
    """Analyze sentiment and narrative for discovered events.

    Real implementation using keyword analysis + optional LLM enrichment.
    """
    events = state.get("events", [])
    logger.info("[sentiment-service] analyzing %d events", len(events))

    if not events:
        return {"sentiment_scores": [], "sentiment_summary": ""}

    # Get news items from state or reconstruct from events
    news_items: list[dict] = []
    for evt in events:
        news_items.append({
            "title": evt.get("title", ""),
            "content": evt.get("content", "") or evt.get("summary", ""),
            "symbols": evt.get("symbol_list", []),
        })

    # Try LLM enrichment
    llm = None
    try:
        from services.llm_adapter.llm_provider import create_quick_llm
        llm = create_quick_llm()
    except Exception:
        pass

    scores = await analyze_news_batch(events, news_items, llm=llm)

    # Build summary
    if scores:
        avg_sentiment = sum(s.sentiment_score for s in scores) / len(scores)
        avg_hype = sum(s.hype_score for s in scores) / len(scores)
        direction = "偏多" if avg_sentiment > 0.15 else ("偏空" if avg_sentiment < -0.15 else "中性")
        summary = (
            f"舆情方向: {direction} | "
            f"平均情绪: {avg_sentiment:+.3f} | "
            f"炒作热度: {avg_hype:.3f} | "
            f"样本数: {len(scores)}"
        )
    else:
        summary = "无足够的舆情数据进行分析"

    score_dicts = [s.model_dump() for s in scores]
    return {"sentiment_scores": score_dicts, "sentiment_summary": summary}
