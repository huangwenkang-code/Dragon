"""Export latest pipeline run as /run API response JSON for frontend debugging."""
import asyncio, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

from db.connection import async_session_factory
from sqlalchemy import select, text
from db.models import (PipelineRun, Event, SentimentScore, LeaderCandidate,
                       ActiveStock, CapitalFlowRecord, DragonTigerRecord,
                       SectorFlowRecord, RiskFlag, ActivatedMemory)


async def main():
    async with async_session_factory() as s:
        q = select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(1)
        r = (await s.execute(q)).scalar_one_or_none()
        if not r:
            print("No runs in database")
            return

        events = (await s.execute(
            select(Event).where(Event.run_id == r.run_id))).scalars().all()
        sentiments = (await s.execute(
            select(SentimentScore).where(SentimentScore.run_id == r.run_id))).scalars().all()
        candidates = (await s.execute(
            select(LeaderCandidate).where(LeaderCandidate.run_id == r.run_id))).scalars().all()
        flows = (await s.execute(
            select(CapitalFlowRecord).where(CapitalFlowRecord.run_id == r.run_id))).scalars().all()
        lhbs = (await s.execute(
            select(DragonTigerRecord).where(DragonTigerRecord.run_id == r.run_id))).scalars().all()
        sectors = (await s.execute(
            select(SectorFlowRecord).where(SectorFlowRecord.run_id == r.run_id))).scalars().all()
        actives = (await s.execute(
            select(ActiveStock).where(ActiveStock.run_id == r.run_id))).scalars().all()
        risks = (await s.execute(
            select(RiskFlag).where(RiskFlag.run_id == r.run_id))).scalars().all()

        # Build RunResponse-format dict
        result = {
            "run_id": r.run_id,
            "status": "completed",
            "events": [],
            "sentiment_scores": [],
            "capital_flow_records": [],
            "sector_flow_records": [],
            "capital_flow_summary": {},
            "active_stocks": [],
            "dragon_tiger_records": [],
            "leader_candidates": [],
            "risk_flags": [],
            "metadata": r.metadata_ or {},
        }

        for e in events:
            stocks_q = text("SELECT stock_code FROM event_stocks WHERE event_id = :eid")
            stocks = (await s.execute(stocks_q, {"eid": e.id})).scalars().all()
            result["events"].append({
                "event_id": e.event_id, "event_type": e.event_type,
                "title": e.title, "summary": e.summary or "",
                "content": e.content or "", "source": e.source or "",
                "publish_time": e.publish_time or "", "narrative": e.narrative or "",
                "event_strength": e.event_strength or 0,
                "heat_score": e.heat_score or 0,
                "strength": e.strength or 0, "novelty": e.novelty or 0,
                "scope": e.scope or "individual",
                "keywords": e.keywords or [], "sector_list": e.sector_list or [],
                "sector_tags": e.sector_tags or [], "symbol_list": list(stocks),
                "llm_prompt": e.llm_prompt or "",
                "llm_response": e.llm_response or "",
                "llm_model": e.llm_model or "",
            })

        for s in sentiments:
            result["sentiment_scores"].append({
                "target_id": s.target_id, "target_type": s.target_type,
                "symbol": s.symbol, "sentiment_score": s.sentiment_score,
                "narrative_score": s.narrative_score, "hype_score": s.hype_score,
                "consistency_score": s.consistency_score, "risk_score": s.risk_score,
                "confidence": s.confidence, "heat": s.heat,
                "consensus": s.consensus, "diffusion_speed": s.diffusion_speed,
                "narrative_strength": s.narrative_strength,
                "keywords": s.keywords or [],
                "finbert_positive": s.finbert_positive or 0,
                "finbert_negative": s.finbert_negative or 0,
                "finbert_neutral": s.finbert_neutral or 0,
                "llm_prompt": s.llm_prompt or "",
                "llm_response": s.llm_response or "",
            })

        for f in flows:
            result["capital_flow_records"].append({
                "symbol": f.symbol, "stock_name": f.stock_name,
                "price": f.price or 0, "change_pct": f.change_pct or 0,
                "amount": f.amount or 0, "amount_wan": f.amount_wan or 0,
                "main_force_net": f.main_force_net or 0,
                "main_force_ratio": f.main_force_ratio or 0,
                "super_large_net": f.super_large_net or 0,
                "large_net": f.large_net or 0, "mid_net": f.mid_net or 0,
                "small_net": f.small_net or 0, "total_net": f.total_net or 0,
                "northbound_net": f.northbound_net or 0,
                "flow_ratio": f.flow_ratio or 0, "sector_flow": f.sector_flow or 0,
                "flow_score": f.flow_score or 0, "pe": f.pe or 0,
                "pb": f.pb or 0, "market_cap": f.market_cap or 0,
                "_source": f.data_source or "",
            })

        for l in lhbs:
            result["dragon_tiger_records"].append({
                "stock_code": l.stock_code, "stock_name": l.stock_name,
                "trade_date": str(l.trade_date) if l.trade_date else "",
                "reason": l.reason or "", "buy_seats": l.buy_seats or [],
                "sell_seats": l.sell_seats or [], "total_buy": l.total_buy or 0,
                "total_sell": l.total_sell or 0, "net_amount": l.net_amount or 0,
                "famous_traders": l.famous_traders or [],
                "trader_signal": l.trader_signal or "", "lhb_score": l.lhb_score or 0,
            })

        for sec in sectors:
            result["sector_flow_records"].append({
                "sector_code": sec.sector_code, "sector_name": sec.sector_name,
                "change_pct": sec.change_pct or 0,
                "turnover_yi": sec.turnover_yi or 0,
                "main_force_net": sec.main_force_net or 0,
                "main_force_ratio": sec.main_force_ratio or 0,
                "super_large_net": sec.super_large_net or 0,
                "large_net": sec.large_net or 0, "heat": sec.heat or 0,
                "stock_count": sec.stock_count or 0,
                "up_count": sec.up_count or 0, "down_count": sec.down_count or 0,
                "leading_stock": sec.leading_stock or "",
                "leading_stock_name": sec.leading_stock_name or "",
                "leading_stock_change": sec.leading_stock_change or 0,
            })

        for a in actives:
            result["active_stocks"].append({
                "symbol": a.symbol, "stock_name": a.stock_name,
                "rank": a.rank, "active_score": a.active_score or 0,
                "flow_score": a.flow_score or 0,
                "concept_score": a.concept_score or 0,
                "lhb_score": a.lhb_score or 0,
                "main_force_net": a.main_force_net or 0,
                "ddejingliang": a.ddejingliang or 0,
                "super_large_net": a.super_large_net or 0,
                "large_net": a.large_net or 0, "mid_net": a.mid_net or 0,
                "small_net": a.small_net or 0, "amount_wan": a.amount_wan or 0,
                "change_pct": a.change_pct or 0, "pe": a.pe or 0,
                "pb": a.pb or 0, "market_cap": a.market_cap or 0,
                "_source": a.data_source or "", "reasons": a.reasons or "",
                "matched_concepts": a.matched_concepts or [],
            })

        for c in candidates:
            result["leader_candidates"].append({
                "stock_code": c.stock_code, "stock_name": c.stock_name,
                "trade_date": str(c.trade_date) if c.trade_date else "",
                "rank": c.rank, "leader_score": c.leader_score,
                "monster_reference": c.monster_reference,
                "monster_potential": c.monster_potential or 0,
                "limit_up_prob": c.limit_up_prob or 0,
                "reasoning": c.reasoning or "", "sector": c.sector or "",
                "sentiment_sub": c.sentiment_sub or 0,
                "flow_sub": c.flow_sub or 0, "lhb_sub": c.lhb_sub or 0,
                "ml_sub": c.ml_sub or 0, "event_sub": c.event_sub or 0,
                "sector_tag_sub": c.sector_tag_sub or 0,
            })

        for risk in risks:
            result["risk_flags"].append({
                "stock_code": risk.stock_code, "risk_type": risk.risk_type,
                "severity": risk.severity or 0,
                "description": risk.description or "",
            })

        # Stats
        stats = {k: len(v) for k, v in result.items()
                 if isinstance(v, list) and k != "metadata"}
        result["_stats"] = stats

        # Save
        out_dir = os.path.join(os.path.dirname(__file__), "..", "dragon-engine-web", "public")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pipeline_result_sample.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        from datetime import datetime
        result["_saved_at"] = datetime.now().isoformat()
        print(f"Saved {out_path}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"File size: {os.path.getsize(out_path):,} bytes")


asyncio.run(main())
