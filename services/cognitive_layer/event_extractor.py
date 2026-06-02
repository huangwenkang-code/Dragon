"""LLM-powered event extractor — extracts structured Event objects from concept-level news.

Two-pass pattern (adapted from Biomni evaluateGPT):
  Pass 1: LLM free-form reasoning — identify which news items are event-worthy
  Pass 2: Structured extraction — parse LLM output into Pydantic Event models

Fallback: keyword-based extraction if LLM is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.language_models import BaseChatModel

from shared.schemas.agent_state import Event
from shared.utils.logging import get_logger
from services.cognitive_layer.prompts.event_extraction import event_extraction_prompt

logger = get_logger(__name__)

# Max news items per batch (avoid LLM context overflow)
_BATCH_SIZE = 15

# Max total events to return (keep focused)
_MAX_EVENTS = 20


async def extract_events_from_news(
    news_articles: list[dict],
    llm: BaseChatModel,
    trade_date: str = "",
    *,
    batch_size: int = _BATCH_SIZE,
) -> list[Event]:
    """Extract structured market events from concept-level news articles using LLM.

    Args:
        news_articles: List of news dicts from fetch_concept_news()
        llm: LangChain chat model (DeepSeek, OpenAI, etc.)
        trade_date: Trading date for context
        batch_size: Max news items per LLM batch

    Returns:
        List of Event objects, ranked by event_strength descending
    """
    if not news_articles:
        logger.info("[event_extractor] no news to extract events from")
        return []

    # Split into batches
    batches = [news_articles[i:i + batch_size] for i in range(0, len(news_articles), batch_size)]
    logger.info("[event_extractor] %d articles → %d batches (batch_size=%d)",
                len(news_articles), len(batches), batch_size)

    # Parallel LLM calls across batches
    tasks = [_extract_batch(b, llm, trade_date, i + 1, len(batches))
             for i, b in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_events: list[Event] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("[event_extractor] batch %d/%d LLM failed: %s",
                         i + 1, len(batches), result)
            fallback = _fallback_extract(batches[i])
            all_events.extend(fallback)
            logger.info("[event_extractor] batch %d/%d: fallback extracted %d events",
                        i + 1, len(batches), len(fallback))
        else:
            all_events.extend(result)
            logger.info("[event_extractor] batch %d/%d: extracted %d events",
                        i + 1, len(batches), len(result))

    # Deduplicate by title similarity
    all_events = _deduplicate_events(all_events)

    # Rank by event_strength and limit
    all_events.sort(key=lambda e: e.event_strength, reverse=True)
    if len(all_events) > _MAX_EVENTS:
        all_events = all_events[:_MAX_EVENTS]

    logger.info("[event_extractor] final: %d unique events (top strength=%.2f)",
                len(all_events), all_events[0].event_strength if all_events else 0)

    return all_events


async def _extract_batch(
    batch: list[dict],
    llm: BaseChatModel,
    trade_date: str,
    batch_num: int,
    total_batches: int,
) -> list[Event]:
    """Extract events from a single batch of news using LLM."""
    # Format news as text
    news_lines = []
    for i, article in enumerate(batch):
        content_preview = article.get("content", "")[:300]
        news_lines.append(
            f"[ID: {i}] 标题: {article.get('title', '')} | "
            f"来源: {article.get('source', '')} | "
            f"时间: {article.get('publish_time', '')} | "
            f"搜索关键词: {article.get('concept_query', '')} | "
            f"内容: {content_preview}"
        )
    news_text = "\n---\n".join(news_lines)

    # Build prompt
    prompt = event_extraction_prompt.invoke({
        "trade_date": trade_date or "今日",
        "news_count": len(batch),
        "news_text": news_text,
        "batch_num": batch_num,
        "total_batches": total_batches,
    })
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    llm_model = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""

    # Call LLM
    response = await llm.ainvoke(prompt)
    raw_response = response.content if hasattr(response, "content") else str(response)

    # Parse JSON from response
    events_data = _parse_json_response(raw_response)

    # Convert to Event models
    events: list[Event] = []
    for ed in events_data:
        try:
            event = Event(
                event_id=ed.get("event_id", f"evt_{batch_num}_{len(events)}"),
                event_type=ed.get("event_type", "题材"),
                title=ed.get("title", ""),
                summary=ed.get("summary", ""),
                content=ed.get("content", ""),
                source=ed.get("source", ""),
                publish_time=ed.get("publish_time", ""),
                symbol_list=ed.get("symbol_list", []),
                sector_list=ed.get("sector_list", []),
                sector_tags=ed.get("sector_tags", []),
                narrative=ed.get("narrative", ""),
                event_strength=float(ed.get("event_strength", 0.5)),
                heat_score=float(ed.get("heat_score", 0.5)),
                keywords=ed.get("keywords", []),
                strength=float(ed.get("strength", ed.get("event_strength", 0.5))),
                novelty=float(ed.get("novelty", 0.5)),
                scope=ed.get("scope", "sector"),
                llm_prompt=prompt_text,
                llm_response=raw_response,
                llm_model=llm_model,
            )
            if event.title:  # Don't add events without titles
                events.append(event)
        except Exception as exc:
            logger.warning("[event_extractor] failed to parse event: %s", exc)

    return events


def _parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response (robust to markdown wrapping, trailing text)."""
    # Strip thinking tags that some models emit
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text)
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text)

    # Try direct JSON parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    for pattern in [r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        match = re.search(pattern, text)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, list):
                    return result
                if isinstance(result, dict):
                    return [result]
            except (json.JSONDecodeError, AttributeError):
                continue

    # Try to find JSON array in the text (greedy: find [...])
    # Handle both empty [] and [...] with content
    array_match = re.search(r"\[([\s\S]*?)\]", text)
    if array_match:
        json_str = array_match.group(0)  # The full [ ... ] match
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            # Try to extract just the first complete array
            bracket_count = 0
            start = -1
            for i, ch in enumerate(text):
                if ch == "[" and bracket_count == 0:
                    start = i
                    bracket_count = 1
                elif ch == "[" and bracket_count > 0:
                    bracket_count += 1
                elif ch == "]":
                    bracket_count -= 1
                    if bracket_count == 0 and start >= 0:
                        json_str = text[start:i + 1]
                        try:
                            result = json.loads(json_str)
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

    logger.warning("[event_extractor] could not parse JSON from LLM response: %s", text[:300])
    return []


def _deduplicate_events(events: list[Event]) -> list[Event]:
    """Remove events with highly similar titles."""
    seen: set[str] = set()
    unique: list[Event] = []
    for e in events:
        # Simple dedup: first 20 chars of title
        key = e.title[:20].strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(e)
        elif not key:
            unique.append(e)  # Don't dedup empty titles (shouldn't happen)
    return unique


# ---------------------------------------------------------------------------
# Keyword-based fallback (no LLM needed)
# ---------------------------------------------------------------------------

def _fallback_extract(news_articles: list[dict]) -> list[Event]:
    """Extract events using keyword matching (no LLM).

    This is the FALLBACK — only used when LLM is unavailable.
    Far less accurate but keeps the pipeline running.
    """
    # Keywords that suggest event-worthy content
    EVENT_INDICATORS = [
        "利好", "涨停", "异动", "爆发", "领涨", "龙头", "热点",
        "政策", "发布", "突破", "签约", "中标", "获批", "重组",
        "增持", "回购", "业绩", "预增", "扭亏",
    ]

    events: list[Event] = []
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        text = title + content[:200]

        # Check if article contains event indicators
        matches = [kw for kw in EVENT_INDICATORS if kw in text]
        if not matches:
            continue

        # Extract stock codes from content (6-digit numbers)
        stock_codes = re.findall(r"\b(00\d{4}|30\d{4}|60\d{4}|68\d{4})\b", content[:500])

        # Score based on indicator count
        strength = min(0.7, len(matches) * 0.15)

        # Determine event type
        event_type = "题材"
        if any(kw in text for kw in ["政策", "国务院", "发改委", "工信部"]):
            event_type = "政策"
        elif any(kw in text for kw in ["业绩", "公告", "股东大会"]):
            event_type = "公告"
        elif any(kw in text for kw in ["突破", "发布", "发布"]):
            event_type = "产业"
        elif any(kw in text for kw in ["涨停", "异动", "爆发"]):
            event_type = "突发"

        event = Event(
            event_id=f"kw_{len(events)}",
            event_type=event_type,
            title=title[:60],
            summary=content[:200],
            content=content[:500],
            source=article.get("source", ""),
            publish_time=article.get("publish_time", ""),
            symbol_list=list(set(stock_codes))[:10],
            sector_list=[article.get("concept_query", "")] if article.get("concept_query") else [],
            keywords=matches[:8],
            event_strength=round(strength, 2),
            heat_score=round(min(0.8, len(matches) * 0.2), 2),
            strength=round(strength, 2),
            novelty=0.3,  # fallback can't assess novelty well
            scope="sector" if len(stock_codes) > 3 else "individual",
        )
        events.append(event)

    return events
