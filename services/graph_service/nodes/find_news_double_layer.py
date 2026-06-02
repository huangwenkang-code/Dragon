"""find_news_double_layer — concept-level + stock-level news for active stocks.

Two layers:
  1. Concept-level: search news by hot concept/sector names → extract sector-driving events
  2. Stock-level: for top active stocks in each event's scope, find connecting news

The key insight: searching by CONCEPT name (e.g., "鸿蒙概念") returns analytical articles
about WHY the sector is moving, whereas searching by stock code returns garbage 快讯.
"""

from __future__ import annotations

from shared.schemas.agent_state import AgentState
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def find_news_double_layer(state: AgentState) -> dict:
    """Find news explaining WHY active stocks and sectors are moving.

    Pipeline:
      1. Extract hot concept/sector names from state (sector_tags + sector_flow)
      2. Search EastMoney by concept name → get analytical sector-level articles
      3. LLM extracts structured events from articles
      4. Events flow downstream → analyze_sentiment → generate_candidates
    """
    watchlist = state.get("watchlist", [])
    active_stocks = state.get("active_stocks", [])
    sector_tags = state.get("sector_tags", [])
    sector_flow_records = state.get("sector_flow_records", [])
    trade_date = state.get("trade_date", "")

    logger.info("[find_news] %d watchlist, %d active, %d sector_tags, %d sector_flow",
                len(watchlist), len(active_stocks), len(sector_tags), len(sector_flow_records))

    # ------------------------------------------------------------------
    # Step 1: Determine WHAT to search for (hot concepts/sectors)
    # ------------------------------------------------------------------
    from services.cognitive_layer.news_fetcher import extract_concept_names

    concept_names = extract_concept_names(sector_tags, sector_flow_records)
    logger.info("[find_news] extracted %d concept names to search: %s",
                len(concept_names), concept_names[:8])

    if not concept_names:
        logger.warning("[find_news] no concept names — using top active stock names as fallback")
        # Fallback: use stock names of top active stocks as search terms
        for s in active_stocks[:10]:
            name = s.get("stock_name", "")
            if name and name not in concept_names:
                concept_names.append(name)

    if not concept_names:
        return {"events": [], "event_count": 0}

    # ------------------------------------------------------------------
    # Step 2: Fetch concept-level news from EastMoney
    # ------------------------------------------------------------------
    from services.cognitive_layer.news_fetcher import fetch_concept_news

    news_articles = await fetch_concept_news(concept_names, trade_date=trade_date)
    if not news_articles:
        logger.warning("[find_news] no news articles found for any concept")
        return {"events": [], "event_count": 0}

    logger.info("[find_news] fetched %d concept-level news articles", len(news_articles))

    # ------------------------------------------------------------------
    # Step 3: LLM extracts structured events from articles
    # ------------------------------------------------------------------
    try:
        from services.llm_adapter.llm_provider import create_deep_llm
        from services.cognitive_layer.event_extractor import extract_events_from_news

        llm = create_deep_llm()  # Use deep-thinking model for event extraction
        events = await extract_events_from_news(news_articles, llm, trade_date)

        logger.info("[find_news] LLM extracted %d events from %d articles",
                    len(events), len(news_articles))

    except Exception as exc:
        logger.warning("[find_news] LLM extraction failed (%s), using keyword fallback", exc)
        from services.cognitive_layer.event_extractor import _fallback_extract
        events = _fallback_extract(news_articles)
        logger.info("[find_news] keyword fallback extracted %d events", len(events))

    # ------------------------------------------------------------------
    # Step 4: Build symbol list for each event (connect events to stocks)
    # ------------------------------------------------------------------
    events = _enrich_event_symbols(events, active_stocks, sector_tags)

    # Convert Event models to dicts for AgentState
    event_dicts = [e.model_dump() for e in events]

    # Log top events for debugging
    for i, e in enumerate(events[:5]):
        logger.info("[find_news] event #%d: [%s] %s (strength=%.2f, symbols=%s)",
                    i + 1, e.event_type, e.title[:60], e.event_strength, e.symbol_list[:5])

    return {
        "events": event_dicts,
        "event_count": len(event_dicts),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich_event_symbols(
    events: list,
    active_stocks: list[dict],
    sector_tags: list[dict],
) -> list:
    """Connect events to specific stocks by matching sector/concept names.

    For each event:
    1. Check if it already has symbol_list from LLM extraction
    2. If not, find active stocks whose sector_tags overlap with event's sector_list
    3. Add those stocks to the event's symbol_list
    """
    # Build sector → stocks index from sector_tags
    sector_to_stocks: dict[str, list[str]] = {}
    for tag in sector_tags:
        concept = tag.get("concept_name", "")
        leader = tag.get("leader_stock", "")
        if concept and leader:
            sector_to_stocks.setdefault(concept, []).append(leader)

    # Build concept → stocks index from active_stocks
    for s in active_stocks:
        for concept in s.get("matched_concepts", []):
            sector_to_stocks.setdefault(concept, []).append(s.get("symbol", ""))

    for event in events:
        if event.symbol_list:
            continue  # Already has symbols from LLM

        # Find stocks in sectors mentioned by this event
        symbols: list[str] = []
        for sector in event.sector_list:
            stocks = sector_to_stocks.get(sector, [])
            for sym in stocks:
                if sym not in symbols:
                    symbols.append(sym)

        if symbols:
            event.symbol_list = symbols[:10]

    return events
