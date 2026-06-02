"""Persist pipeline result dict to PostgreSQL.

Single-transaction write of all 14 tables from the graph.ainvoke() result.
Usage:
    from db.persist import persist_run
    await persist_run(trade_date, result_dict)
"""

from datetime import datetime

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import async_session_factory
from db.models import (ActivatedMemory, ActiveStock, CapitalFlowRecord,
                       DragonTigerRecord, Event, EventStock, LeaderCandidate,
                       PipelineRun, RiskFlag, SectorConcept, SectorFlowRecord,
                       SentimentScore, StockConceptTag)
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def persist_run(trade_date: str, result: dict) -> str:
    """Persist a full pipeline run result. Returns run_id."""
    run_id = result.get("metadata", {}).get("started_at") or datetime.utcnow().isoformat()

    trade_date_obj = datetime.strptime(trade_date[:10], "%Y-%m-%d").date()

    async with async_session_factory() as session:
        async with session.begin():
            # ---- 0. Dedup: delete ALL existing runs for same trade_date ----
            old_runs = (await session.execute(
                select(PipelineRun).where(PipelineRun.trade_date == trade_date_obj)
            )).scalars().all()
            for old in old_runs:
                await session.delete(old)
            if old_runs:
                await session.flush()
                logger.info("[persist] removed %d previous run(s) for %s", len(old_runs), trade_date_obj)

            # ---- 1. pipeline_run ----
            events = result.get("events", [])
            candidates = result.get("leader_candidates", [])
            top_score = max((c.get("leader_score", 0) for c in candidates), default=0)

            run = PipelineRun(
                run_id=run_id,
                trade_date=datetime.strptime(trade_date[:10], "%Y-%m-%d").date(),
                status="completed",
                watchlist=result.get("watchlist", []),
                top_n=result.get("top_n", 5),
                event_count=len(events),
                candidate_count=len(candidates),
                top_score=round(top_score, 4),
                errors=result.get("metadata", {}).get("errors", []),
                metadata_=result.get("metadata", {}),
            )
            session.add(run)
            await session.flush()  # ensure run_id visible for FK references

            # Save token usage from tracker
            try:
                from services.token_tracker import TokenUsageTracker
                tracker = TokenUsageTracker.instance()
                summary = tracker.summary()
                logger.info("[persist] token tracker summary: tokens=%d cost=%.6f records=%d",
                            summary["total_tokens"], summary["total_cost"], len(summary["records"]))
                if summary["total_tokens"] > 0:
                    run.token_usage = {
                        "total_cost": summary["total_cost"],
                        "total_tokens": summary["total_tokens"],
                        "total_prompt_tokens": summary["total_prompt_tokens"],
                        "total_completion_tokens": summary["total_completion_tokens"],
                        "records": summary["records"],
                    }
                else:
                    logger.warning("[persist] token tracker has 0 total_tokens — no LLM calls tracked?")
                    run.token_usage = summary  # at least save empty structure for diagnosis
            except Exception:
                logger.exception("[persist] token tracker unavailable")

            # ---- 2. events + event_stocks ----
            for evt in events:
                orm_evt = Event(
                    run_id=run_id,
                    event_id=evt.get("event_id", evt.get("id", "")),
                    event_type=evt.get("event_type", ""),
                    title=evt.get("title", ""),
                    summary=evt.get("summary", ""),
                    content=evt.get("content", ""),
                    source=evt.get("source", ""),
                    publish_time=evt.get("publish_time", ""),
                    narrative=evt.get("narrative", ""),
                    event_strength=evt.get("event_strength", 0),
                    heat_score=evt.get("heat_score", 0),
                    strength=evt.get("strength", 0),
                    novelty=evt.get("novelty", 0),
                    scope=evt.get("scope", "individual"),
                    keywords=evt.get("keywords", []),
                    sector_list=evt.get("sector_list", []),
                    sector_tags=evt.get("sector_tags", []),
                    llm_prompt=evt.get("llm_prompt", ""),
                    llm_response=evt.get("llm_response", ""),
                    llm_model=evt.get("llm_model", ""),
                )
                session.add(orm_evt)
                await session.flush()

                for sym in evt.get("symbol_list", []):
                    session.add(EventStock(event_id=orm_evt.id, stock_code=sym))

            # ---- 3. sentiment_scores ----
            for s in result.get("sentiment_scores", []):
                session.add(SentimentScore(
                    run_id=run_id,
                    target_id=s.get("target_id", ""),
                    target_type=s.get("target_type", "stock"),
                    symbol=s.get("symbol", ""),
                    sentiment_score=s.get("sentiment_score", 0),
                    narrative_score=s.get("narrative_score", s.get("narrative_strength", 0)),
                    hype_score=s.get("hype_score", 0),
                    consistency_score=s.get("consistency_score", s.get("consensus", 0.5)),
                    risk_score=s.get("risk_score", 0),
                    confidence=s.get("confidence", 0.5),
                    heat=s.get("heat", 0),
                    consensus=s.get("consensus", s.get("consistency_score", 0.5)),
                    diffusion_speed=s.get("diffusion_speed", 0),
                    narrative_strength=s.get("narrative_strength", s.get("narrative_score", 0)),
                    keywords=s.get("keywords", []),
                    finbert_positive=s.get("finbert_positive", 0),
                    finbert_negative=s.get("finbert_negative", 0),
                    finbert_neutral=s.get("finbert_neutral", 0),
                    llm_prompt=s.get("llm_prompt", ""),
                    llm_response=s.get("llm_response", ""),
                ))

            # ---- 4. capital_flow_records ----
            for r in result.get("capital_flow_records", []):
                session.add(CapitalFlowRecord(
                    run_id=run_id,
                    symbol=r.get("symbol", ""),
                    stock_name=r.get("stock_name", ""),
                    price=r.get("price", 0),
                    change_pct=r.get("change_pct", 0),
                    amount=r.get("amount", 0),
                    amount_wan=r.get("amount_wan", 0),
                    main_force_net=r.get("main_force_net", 0),
                    main_force_ratio=r.get("main_force_ratio", 0),
                    super_large_net=r.get("super_large_net", 0),
                    large_net=r.get("large_net", 0),
                    mid_net=r.get("mid_net", 0),
                    small_net=r.get("small_net", 0),
                    total_net=r.get("total_net", 0),
                    northbound_net=r.get("northbound_net", 0),
                    flow_ratio=r.get("flow_ratio", 0),
                    sector_flow=r.get("sector_flow", 0),
                    flow_score=r.get("flow_score", 0),
                    pe=r.get("pe", 0),
                    pb=r.get("pb", 0),
                    market_cap=r.get("market_cap", 0),
                    turnover_pct=r.get("turnover_pct", 0),
                    data_source=r.get("_source", r.get("data_source", "")),
                ))

            # ---- 5. sector_flow_records ----
            for s in result.get("sector_flow_records", []):
                session.add(SectorFlowRecord(
                    run_id=run_id,
                    sector_code=s.get("sector_code", ""),
                    sector_name=s.get("sector_name", ""),
                    change_pct=s.get("change_pct", 0),
                    turnover_yi=s.get("turnover_yi", 0),
                    main_force_net=s.get("main_force_net", 0),
                    main_force_ratio=s.get("main_force_ratio", 0),
                    super_large_net=s.get("super_large_net", 0),
                    large_net=s.get("large_net", 0),
                    heat=s.get("heat", 0),
                    stock_count=s.get("stock_count", 0),
                    up_count=s.get("up_count", 0),
                    down_count=s.get("down_count", 0),
                    leading_stock=s.get("leading_stock", ""),
                    leading_stock_name=s.get("leading_stock_name", ""),
                    leading_stock_change=s.get("leading_stock_change", 0),
                ))

            # ---- 6. dragon_tiger_records ----
            for r in result.get("dragon_tiger_records", []):
                session.add(DragonTigerRecord(
                    run_id=run_id,
                    stock_code=r.get("stock_code", ""),
                    stock_name=r.get("stock_name", ""),
                    trade_date=datetime.strptime(str(r.get("trade_date", trade_date[:10]))[:10], "%Y-%m-%d").date(),
                    reason=r.get("reason", ""),
                    buy_seats=r.get("buy_seats", []),
                    sell_seats=r.get("sell_seats", []),
                    total_buy=r.get("total_buy", 0),
                    total_sell=r.get("total_sell", 0),
                    net_amount=r.get("net_amount", 0),
                    famous_traders=r.get("famous_traders", []),
                    trader_signal=r.get("trader_signal", ""),
                    lhb_score=r.get("lhb_score", 0),
                ))

            # ---- 7. active_stocks ----
            for a in result.get("active_stocks", []):
                session.add(ActiveStock(
                    run_id=run_id,
                    symbol=a.get("symbol", ""),
                    stock_name=a.get("stock_name", ""),
                    rank=a.get("rank", 0),
                    active_score=a.get("active_score", 0),
                    flow_score=a.get("flow_score", 0),
                    concept_score=a.get("concept_score", 0),
                    lhb_score=a.get("lhb_score", 0),
                    main_force_net=a.get("main_force_net", 0),
                    ddejingliang=a.get("ddejingliang", 0),
                    super_large_net=a.get("super_large_net", 0),
                    large_net=a.get("large_net", 0),
                    mid_net=a.get("mid_net", 0),
                    small_net=a.get("small_net", 0),
                    amount_wan=a.get("amount_wan", 0),
                    change_pct=a.get("change_pct", 0),
                    pe=a.get("pe", 0),
                    pb=a.get("pb", 0),
                    market_cap=a.get("market_cap", 0),
                    data_source=a.get("_source", ""),
                    reasons=a.get("reasons", ""),
                    matched_concepts=a.get("matched_concepts", []),
                ))

            # ---- 8. sector_concepts + stock_concept_tags ----
            # sector_concepts — use ON CONFLICT DO NOTHING (first writer wins per day)
            concept_rows: list[dict] = []
            for tag in result.get("sector_tags", []):
                if not isinstance(tag, dict):
                    continue
                concept_name = tag.get("concept_name", "")
                if not concept_name:
                    continue
                concept_rows.append(dict(
                    run_id=run_id,
                    concept_name=concept_name,
                    concept_id=str(tag.get("concept_id", "")),
                    leader_stock=tag.get("leader_stock", ""),
                    leader_stock_name=tag.get("leader_stock_name", ""),
                    leader_stock_change=tag.get("leader_stock_change", 0),
                    change_pct=tag.get("change_pct", 0),
                    heat=tag.get("heat", 0),
                    stock_count=tag.get("stock_count", 0),
                    snapshot_date=trade_date_obj,
                ))
            if concept_rows:
                stmt = pg_insert(SectorConcept).values(concept_rows).on_conflict_do_nothing()
                await session.execute(stmt)

            # stock_concept_tags — ON CONFLICT DO NOTHING
            seen_tags: set[tuple] = set()
            tag_rows: list[dict] = []
            for evt in events:
                sym_list = evt.get("symbol_list", [])
                evt_tags = evt.get("sector_tags", [])
                for sym in sym_list:
                    for tag_name in evt_tags:
                        key = (sym, str(tag_name))
                        if key in seen_tags:
                            continue
                        seen_tags.add(key)
                        tag_rows.append(dict(
                            run_id=run_id,
                            stock_code=sym,
                            concept_name=str(tag_name),
                            is_leader=False,
                            snapshot_date=trade_date_obj,
                        ))
            if tag_rows:
                stmt = pg_insert(StockConceptTag).values(tag_rows).on_conflict_do_nothing()
                await session.execute(stmt)

            # ---- 9. leader_candidates ----
            mr_count = sum(1 for c in candidates if c.get("monster_reference") is not None)
            logger.info("[persist] %d candidates, %d with monster_reference",
                        len(candidates), mr_count)
            for c in candidates:
                # Extract sub-scores from the composite formula (approximate breakdown)
                session.add(LeaderCandidate(
                    run_id=run_id,
                    stock_code=c.get("stock_code", ""),
                    stock_name=c.get("stock_name", ""),
                    trade_date=trade_date_obj,
                    rank=c.get("rank", 0),
                    leader_score=c.get("leader_score", 0),
                    monster_potential=c.get("monster_potential", 0),
                    limit_up_prob=c.get("limit_up_prob", 0),
                    reasoning=c.get("reasoning", ""),
                    sector=c.get("sector", ""),
                    sentiment_sub=c.get("sentiment_sub", 0),
                    flow_sub=c.get("flow_sub", 0),
                    lhb_sub=c.get("lhb_sub", 0),
                    ml_sub=c.get("ml_sub", 0),
                    event_sub=c.get("event_sub", 0),
                    sector_tag_sub=c.get("sector_tag_sub", 0),
                    monster_reference=c.get("monster_reference"),
                ))

            # ---- 10. risk_flags ----
            for r in result.get("risk_flags", []):
                session.add(RiskFlag(
                    run_id=run_id,
                    stock_code=r.get("stock_code", ""),
                    risk_type=r.get("risk_type", ""),
                    severity=r.get("severity", 0),
                    description=r.get("description", ""),
                ))

            # ---- 11. activated_memories ----
            for m in result.get("activated_memories", []):
                session.add(ActivatedMemory(
                    run_id=run_id,
                    current_event_title=m.get("current_event_title", ""),
                    historical_summary=m.get("historical_summary", ""),
                    similarity=m.get("similarity", 0),
                    lifecycle_stage=m.get("lifecycle_stage", ""),
                    memory_created_at=m.get("created_at", ""),
                ))

        # commit on exit of async with session.begin()
        total_rows = (
            len(events) + len(result.get("sentiment_scores", [])) +
            len(result.get("capital_flow_records", [])) +
            len(result.get("sector_flow_records", [])) +
            len(result.get("dragon_tiger_records", [])) +
            len(result.get("active_stocks", [])) +
            len(candidates)
        )
        logger.info("[persist] run_id=%s trade_date=%s rows=%d", run_id, trade_date, total_rows)

    return run_id


async def upsert_daily_bars(session: AsyncSession, bars: list[dict]) -> int:
    """Upsert daily OHLCV bars. Returns count of rows inserted/updated."""
    if not bars:
        return 0
    from db.models import StockDailyBar

    from datetime import date as date_t
    rows = []
    for b in bars:
        td = b.get("trade_date")
        if isinstance(td, str):
            td = date_t.fromisoformat(td[:10])
        rows.append(dict(
            symbol=b.get("symbol", ""),
            trade_date=td,
            open=b.get("open", 0),
            high=b.get("high", 0),
            low=b.get("low", 0),
            close=b.get("close", 0),
            volume=b.get("volume", 0),
            amount=b.get("amount", 0),
            change_pct=b.get("change_pct", 0),
            turnover_pct=b.get("turnover_pct", 0),
        ))

    stmt = pg_insert(StockDailyBar).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "trade_date"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "amount": stmt.excluded.amount,
            "change_pct": stmt.excluded.change_pct,
            "turnover_pct": stmt.excluded.turnover_pct,
        }
    )
    await session.execute(stmt)
    return len(rows)
