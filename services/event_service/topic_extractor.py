"""Topic extraction via LLM — identifies themes, sectors, and event types."""

from __future__ import annotations

import json
import re

from shared.schemas.agent_state import Event
from shared.utils.logging import get_logger

logger = get_logger(__name__)

TOPIC_EXTRACTION_PROMPT = """你是一位A股市场事件分析专家。请分析以下新闻，提取关键信息。

新闻标题：{title}
新闻内容：{content}

请输出JSON格式（不要输出其他内容）：
{{
  "event_type": "政策/产业/公告/突发/题材",
  "sector_list": ["板块1", "板块2"],
  "narrative": "一句话概括这个事件的市场叙事",
  "event_strength": 0.0-1.0之间的数字，表示事件对市场的冲击力,
  "keywords": ["关键词1", "关键词2"]
}}

事件强度判断标准：
- 0.8-1.0: 重大政策/突发黑天鹅/行业拐点级别
- 0.5-0.7: 一般政策/业绩公告/行业利好
- 0.2-0.4: 个股新闻/小范围影响
- 0.0-0.1: 普通报道/无实质影响
"""


def _parse_topic_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling common formatting issues."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON block
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


async def extract_topics(
    news_items: list[dict],
    llm,
    *,
    batch_size: int = 10,
) -> list[Event]:
    """Extract topics from news items using LLM.

    Args:
        news_items: list of news dicts with 'title' and 'content'
        llm: LangChain chat model
        batch_size: max items to process (cost control)

    Returns:
        list of Event objects
    """
    events: list[Event] = []
    items = news_items[:batch_size]

    for i, item in enumerate(items):
        title = item.get("title", "")
        content = item.get("content", "")[:1500]  # truncate long content
        if not title:
            continue

        prompt = TOPIC_EXTRACTION_PROMPT.format(title=title, content=content or title)
        try:
            response = await llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_topic_json(raw_text)

            event_id = f"evt-{i:04d}"
            symbols = item.get("symbols", [])
            event = Event(
                id=event_id,
                event_id=event_id,
                event_type=parsed.get("event_type", "题材"),
                title=title,
                content=content or title,
                summary=parsed.get("narrative", title),
                source=item.get("source", ""),
                publish_time=item.get("publish_time", ""),
                symbol_list=symbols,
                sector_list=parsed.get("sector_list", []),
                sector_tags=parsed.get("sector_list", []),
                narrative=parsed.get("narrative", ""),
                event_strength=min(1.0, max(0.0, float(parsed.get("event_strength", 0.3)))),
                strength=min(1.0, max(0.0, float(parsed.get("event_strength", 0.3)))),
                heat_score=0.0,
                novelty=0.5,
                scope="sector" if len(parsed.get("sector_list", [])) > 1 else "individual",
                keywords=parsed.get("keywords", []),
                timestamp=item.get("publish_time", ""),
            )
            events.append(event)
        except Exception as exc:
            logger.warning("topic extraction failed for '%s': %s", title[:50], exc)
            continue

    logger.info("extract_topics: %d items → %d events", len(items), len(events))
    return events


def extract_topics_sync(
    news_items: list[dict],
    *,
    batch_size: int = 10,
) -> list[Event]:
    """Synchronous fallback — keyword-based topic extraction without LLM."""
    events: list[Event] = []
    items = news_items[:batch_size]

    # Keyword-based classification (no LLM required)
    positive_kw = ["利好", "上涨", "增长", "突破", "涨停", "中标", "签约", "获批", "回购"]
    negative_kw = ["利空", "下跌", "亏损", "暴跌", "处罚", "违规", "退市", "诉讼"]
    policy_kw = ["政策", "国务院", "央行", "证监会", "监管", "发改委", "工信部"]

    for i, item in enumerate(items):
        title = item.get("title", "")
        content = item.get("content", "") or title
        text = f"{title} {content[:500]}"

        # Determine event type
        if any(k in text for k in policy_kw):
            event_type = "政策"
        elif "公告" in text or "业绩" in text or "财报" in text:
            event_type = "公告"
        elif "突发" in text or "紧急" in text:
            event_type = "突发"
        elif any(k in text for k in positive_kw + negative_kw):
            event_type = "产业"
        else:
            event_type = "题材"

        # Compute strength
        strength = 0.3
        pos_count = sum(1 for k in positive_kw if k in text)
        neg_count = sum(1 for k in negative_kw if k in text)
        pol_count = sum(1 for k in policy_kw if k in text)
        strength = min(1.0, 0.3 + 0.1 * (pos_count + neg_count) + 0.15 * pol_count)

        # Extract keywords
        all_kw = positive_kw + negative_kw + policy_kw + ["行业", "市场", "龙头", "题材", "赛道"]
        keywords = list(set(k for k in all_kw if k in text))[:10]

        event_id = f"evt-{i:04d}"
        symbols = item.get("symbols", [])
        event = Event(
            id=event_id,
            event_id=event_id,
            event_type=event_type,
            title=title,
            content=content,
            summary=title,
            source=item.get("source", ""),
            publish_time=item.get("publish_time", ""),
            symbol_list=symbols,
            sector_list=[],
            sector_tags=[],
            narrative=title,
            event_strength=strength,
            strength=strength,
            heat_score=0.0,
            novelty=0.5,
            scope="individual",
            keywords=keywords,
            timestamp=item.get("publish_time", ""),
        )
        events.append(event)

    logger.info("extract_topics_sync: %d items → %d events", len(items), len(events))
    return events
