"""ingest_event node — event extraction + ChromaDB memory lifecycle."""

from shared.schemas.agent_state import AgentState
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def ingest_event(state: AgentState) -> dict:
    """Ingest market events and manage event memory lifecycle.

    1. Fetch news + extract topics + rank (event-service)
    2. Store new events in ChromaDB memory
    3. Query for historically similar active events
    4. Decay old events whose lifecycle is fading
    """
    from services.event_service.extractor import extract_events

    result = await extract_events(state)
    event_count = result.get("event_count", 0)
    logger.info("[ingest_event] extracted %d real events", event_count)

    events = result.get("events", [])
    if not events:
        return {**result, "activated_memories": []}

    # -- ChromaDB event memory integration --
    try:
        from services.memory_service.event_memory import EventMemory, LifecycleStage

        memory = EventMemory(
            name="dragon_engine_events",
            config={"llm_provider": "openai"},
        )

        # Store new events
        triples = []
        for e in events:
            text = _make_event_text(e)
            summary = e.get("summary", e.get("title", ""))[:200]
            triples.append((text, summary, {
                "source": e.get("source", ""),
                "confidence": e.get("event_strength", 0.5),
                "tags": e.get("keywords", []),
                "symbols": e.get("symbol_list", []),
                "lifecycle_stage": LifecycleStage.EMERGING,
                "decay_factor": 1.0,
            }))
        memory.add_events(triples)
        logger.info("[ingest_event] stored %d events in ChromaDB memory", len(triples))

        # Query for historically similar active events
        activated_memories: list[dict] = []
        for e in events[:5]:  # top 5 events only to limit API calls
            query_text = _make_event_text(e)
            similar = memory.query_active_events(query_text, n_matches=3)
            for s in similar:
                if s.get("similarity", 0) > 0.75:
                    activated_memories.append({
                        "current_event_title": e.get("title", ""),
                        "historical_summary": s.get("summary", ""),
                        "similarity": s.get("similarity", 0),
                        "lifecycle_stage": s.get("lifecycle_stage", ""),
                        "created_at": s.get("created_at", ""),
                    })

        if activated_memories:
            logger.info("[ingest_event] activated %d historical memories", len(activated_memories))

    except Exception as exc:
        logger.warning("[ingest_event] ChromaDB memory skipped: %s", exc)
        activated_memories = []

    # -- 同花顺 sector tag enrichment --
    sector_tags: list[dict] = []
    try:
        from shared.data_sources.manager import get_manager
        from shared.data_sources import DataQuery, QueryType
        import time as _time

        manager = get_manager()
        watchlist = state.get("watchlist", [])

        if watchlist:
            _t0 = _time.time()
            logger.info("[ingest_event] calling ths_hot SECTOR_TAGS for %d symbols ...", len(watchlist))
            sector_result = await manager.query(
                "ths_hot",
                DataQuery(QueryType.SECTOR_TAGS, symbols=watchlist),
            )
            _elapsed = _time.time() - _t0
            logger.info("[ingest_event] ths_hot SECTOR_TAGS returned in %.1fs, is_error=%s, data_len=%d",
                        _elapsed, sector_result.is_error, len(sector_result.data) if sector_result.data else 0)
            if not sector_result.is_error and sector_result.data:
                sector_tags = sector_result.data
                # Log first 2 rows to inspect structure
                for i, row in enumerate(sector_tags[:2]):
                    logger.info("[ingest_event] sector_tag[%d] keys=%s concepts=%s",
                                i, list(row.keys()), row.get("concepts", "N/A"))
                # Attach sector tags to each matching event
                for event in events:
                    sym_list = event.get("symbol_list", [])
                    for tag_row in sector_tags:
                        for sym in sym_list:
                            if tag_row.get("symbol") == sym:
                                concepts = tag_row.get("concepts", [])
                                event.setdefault("sector_tags", [])
                                event["sector_tags"].extend(
                                    [c.get("concept_name", "") for c in concepts]
                                )
                logger.info("[ingest_event] enriched %d events with sector tags from %d stock tag sets",
                            len(events), len(sector_tags))
    except Exception as exc:
        logger.warning("[ingest_event] sector tag enrichment skipped: %s", exc)

    return {**result, "activated_memories": activated_memories, "sector_tags": sector_tags}


def _make_event_text(event: dict) -> str:
    """Build a searchable text representation of an event."""
    parts = []
    if event.get("title"):
        parts.append(event["title"])
    if event.get("summary"):
        parts.append(event["summary"])
    if event.get("content"):
        parts.append(event["content"][:500])
    kw = event.get("keywords", [])
    if kw:
        parts.append("关键词: " + ", ".join(kw))
    return "\n".join(parts)
