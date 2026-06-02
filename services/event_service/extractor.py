"""Event extractor — orchestrates news fetching, topic extraction, and ranking."""

from __future__ import annotations

from shared.schemas.agent_state import AgentState, Event
from shared.utils.logging import get_logger
from services.event_service.fetch_news import fetch_all_news
from services.event_service.topic_extractor import extract_topics, extract_topics_sync
from services.event_service.ranking import rank_events

logger = get_logger(__name__)


async def extract_events(state: AgentState) -> dict:
    """Extract market-driving events for all stocks in the watchlist.

    1. Fetch real news from EastMoney / AKShare
    2. Extract topics via LLM (with keyword fallback)
    3. Rank events by composite score
    """
    trade_date = state.get("trade_date", "")
    watchlist = state.get("watchlist", [])

    logger.info("[event-service] scanning %d stocks for %s", len(watchlist), trade_date)

    if not watchlist:
        logger.warning("[event-service] empty watchlist, returning no events")
        return {"events": [], "event_count": 0}

    # Step 1: Fetch real news
    news_items = fetch_all_news(watchlist, max_pages=2, limit_per_stock=30)
    if not news_items:
        logger.warning("[event-service] no news fetched from any source")
        return {"events": [], "event_count": 0}

    logger.info("[event-service] fetched %d news items total", len(news_items))

    # Step 2: Extract topics (try LLM, fallback to keyword)
    events: list[Event] = []
    try:
        from services.llm_adapter.llm_provider import create_quick_llm

        llm = create_quick_llm()
        events = await extract_topics(news_items, llm, batch_size=15)
    except Exception as exc:
        logger.warning("[event-service] LLM unavailable (%s), using keyword fallback", exc)
        events = extract_topics_sync(news_items, batch_size=15)

    if not events:
        return {"events": [], "event_count": 0}

    # Step 3: Rank events
    ranked = rank_events(events, top_n=30)

    # Convert to dict for AgentState (TypedDict uses list[dict])
    event_dicts = [e.model_dump() for e in ranked]

    logger.info("[event-service] produced %d ranked events", len(event_dicts))
    return {"events": event_dicts, "event_count": len(event_dicts)}
