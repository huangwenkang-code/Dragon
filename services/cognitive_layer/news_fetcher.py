"""Concept/sector-level news fetcher — searches EastMoney by concept names.

KEY DESIGN DECISION: We search by CONCEPT/SECTOR name (e.g., "鸿蒙概念", "半导体板块")
rather than by stock code. This returns analytical articles about the sector's movement,
not garbage individual-stock price-broadcast noise.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import httpx

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# EastMoney search API
_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"

# Minimum content length to consider an article "analytical" (not a short flash)
_MIN_CONTENT_LENGTH = 80

# Max articles per concept to avoid overwhelming the LLM
_MAX_PER_CONCEPT = 8

# Max total concepts to search (top N hottest)
_MAX_CONCEPTS = 12


async def fetch_concept_news(
    concept_names: list[str],
    *,
    max_per_concept: int = _MAX_PER_CONCEPT,
    max_total: int = 80,
    http_timeout: float = 15.0,
    trade_date: str = "",
) -> list[dict]:
    """Fetch news articles by searching for concept/sector names.

    Args:
        concept_names: List of concept/sector names to search for
        max_per_concept: Max articles per concept
        max_total: Max total articles across all concepts
        trade_date: If set, filter out articles published after this date (YYYY-MM-DD)

    Returns:
        List of news dicts with keys: title, content, source, url, publish_time, concept_query
    """
    if not concept_names:
        logger.warning("[news_fetcher] no concept names to search")
        return []

    # Limit to hottest concepts to avoid excessive API calls
    concepts = concept_names[:_MAX_CONCEPTS]
    logger.info("[news_fetcher] searching %d concepts (trade_date=%s): %s",
                len(concepts), trade_date or "today", concepts[:5])

    all_articles: list[dict] = []
    seen_titles: set[str] = set()

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        tasks = []
        for concept in concepts:
            tasks.append(_search_concept(client, concept, max_per_concept))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    for concept, result in zip(concepts, results):
        if isinstance(result, Exception):
            logger.warning("[news_fetcher] concept '%s' search failed: %s", concept, result)
            continue
        for article in result:
            title = article.get("title", "")
            key = title[:80]
            if key not in seen_titles:
                seen_titles.add(key)
                article["concept_query"] = concept
                all_articles.append(article)

    # Filter by trade_date: discard articles published after the target date
    if trade_date:
        before_count = len(all_articles)
        all_articles = [a for a in all_articles if a.get("publish_time", "")[:10] <= trade_date]
        logger.info("[news_fetcher] date filter %s: %d → %d articles",
                    trade_date, before_count, len(all_articles))

    # Sort by publish time (newest first) and limit
    all_articles.sort(key=lambda a: a.get("publish_time", ""), reverse=True)
    if len(all_articles) > max_total:
        all_articles = all_articles[:max_total]

    # Log quality stats
    analytical = sum(1 for a in all_articles if len(a.get("content", "")) >= _MIN_CONTENT_LENGTH)
    logger.info(
        "[news_fetcher] %d concepts → %d unique articles (%d analytical, %d short)",
        len(concepts), len(all_articles), analytical, len(all_articles) - analytical,
    )

    return all_articles


async def _search_concept(
    client: httpx.AsyncClient,
    concept: str,
    max_articles: int,
) -> list[dict]:
    """Search EastMoney for news about a specific concept/sector."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.eastmoney.com/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    params = {
        "cb": "jQuery",
        "param": json.dumps({
            "uid": "",
            "keyword": concept,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": max_articles + 5,  # Fetch extra, filter after
                    "preTag": "",
                    "postTag": "",
                }
            },
        }),
        "_": str(int(time.time() * 1000)),
    }

    try:
        resp = await client.get(_SEARCH_URL, params=params, headers=headers)
        if resp.status_code != 200:
            logger.warning("[news_fetcher] HTTP %d for concept '%s'", resp.status_code, concept)
            return []

        # Strip JSONP wrapper
        text = resp.text
        if text.startswith("jQuery("):
            json_str = text[text.index("(") + 1:text.rindex(")")]
        else:
            json_str = text

        data = json.loads(json_str)
        articles = data.get("result", {}).get("cmsArticleWebOld", [])

        if not articles:
            return []

        parsed = []
        for a in articles:
            title = _clean_html(str(a.get("title", "")))
            content = _clean_html(str(a.get("content", a.get("summary", ""))))
            if not title:
                continue

            parsed.append({
                "title": title,
                "content": content,
                "source": str(a.get("mediaName", a.get("source", "东方财富"))),
                "url": str(a.get("url", "")),
                "publish_time": str(a.get("date", "")),
            })

        # Filter: prefer analytical articles (longer content)
        analytical = [p for p in parsed if len(p["content"]) >= _MIN_CONTENT_LENGTH]
        short = [p for p in parsed if len(p["content"]) < _MIN_CONTENT_LENGTH]

        # Take analytical first, then fill with short (up to max)
        result = analytical[:max_articles]
        remaining = max_articles - len(result)
        if remaining > 0:
            result.extend(short[:remaining])

        logger.debug("[news_fetcher] concept '%s': %d total → %d selected (%d analytical)",
                     concept, len(parsed), len(result), min(len(analytical), max_articles))

        return result

    except Exception as exc:
        logger.warning("[news_fetcher] search failed for '%s': %s", concept, exc)
        return []


def _clean_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Utility: extract concept names from sector_flow state
# ---------------------------------------------------------------------------

def extract_concept_names(
    sector_tags: list[dict],
    sector_flow_records: list[dict],
    top_n: int = _MAX_CONCEPTS,
) -> list[str]:
    """Extract the most important concept/sector names to search for news.

    Priority:
    1. Hot concept names from 同花顺热点 (with highest heat/stock_count)
    2. Top industry sector names by net capital inflow
    """
    names: list[str] = []

    # From sector_tags (concepts with leader stocks — these are hot)
    for tag in sector_tags:
        name = tag.get("concept_name", "").strip()
        if name and name not in names:
            names.append(name)

    # From sector_flow_records (industries by net inflow)
    for rec in sector_flow_records:
        name = rec.get("sector_name", "").strip()
        if name and name not in names:
            names.append(name)

    return names[:top_n]
