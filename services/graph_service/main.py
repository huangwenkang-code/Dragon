"""graph-service FastAPI entry point.

Serve the LangGraph pipeline via a minimal REST API.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so "from services.xxx" imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before any module reads os.getenv
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.graph_service.graph import get_graph
from db.persist import persist_run
from shared.data_sources.bootstrap import bootstrap_sources
from shared.utils.logging import get_logger
from shared.utils.trading_day import get_last_trading_day, get_trading_date_str, is_trading_day

from services.backtest.registry import get_registry
from services.backtest.engine import BacktestEngine
from services.backtest.strategies import TradingStrategy
from services.backtest.reflection import analyze as reflect_analyze

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — warm up the graph and data sources on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Compiling graph ...")
    get_graph()
    logger.info("Graph ready.")
    sources = bootstrap_sources()
    logger.info("Data sources registered: %s", sources)
    from db import init_db
    await init_db()
    logger.info("Database tables verified.")
    yield


app = FastAPI(title="dragon-engine graph-service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    trade_date: str = ""
    top_n: int = 5
    force: bool = False  # If True, run full pipeline (LLM) when no cache; if False, return no_data


class RunResponse(BaseModel):
    run_id: str = ""
    status: str = ""
    events: list = []
    sentiment_scores: list = []
    capital_flow_records: list = []
    sector_flow_records: list = []
    capital_flow_summary: dict = {}
    active_stocks: list = []
    dragon_tiger_records: list = []
    leader_candidates: list = []
    risk_flags: list = []
    metadata: dict = {}


class StrategyConfigRequest(BaseModel):
    name: str
    description: str = ""
    entry_rules: list = []
    exit_rules: list = []
    allocator: dict = {}
    max_positions: int = 999
    max_position_pct: float = 1.0
    initial_capital: float = 100000.0
    daily_cash_pct: float = 0.5
    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    min_commission: float = 5.0
    gap_up_pct: float | None = None
    enable_limit_up_filter: bool = True
    is_system: bool = False


class BacktestRequest(BaseModel):
    strategy_name: str = "A"
    start_date: str = ""
    end_date: str = ""
    use_scanner: bool = False
    scanner_top_n: int = 60
    use_v4: bool = True  # V4 5-factor scoring (default)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "graph-service", "timestamp": datetime.now().isoformat()}


async def _load_run_from_db(trade_date: str) -> dict | None:
    """Load a previously-persisted pipeline run from DB. Returns None if not found."""
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun, Event, EventStock, LeaderCandidate, SentimentScore
    from db.models import CapitalFlowRecord, SectorFlowRecord, RiskFlag, DragonTigerRecord

    trade_date_obj = datetime.strptime(trade_date[:10], "%Y-%m-%d").date()

    async with async_session_factory() as session:
        # Find the most recent run for this date
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date == trade_date_obj,
                PipelineRun.status == "completed",
            ).order_by(PipelineRun.created_at.desc()).limit(1)
        )
        run = result.scalars().first()
        if not run:
            return None

        run_id = run.run_id
        logger.info("[/run] found cached run_id=%s for %s", run_id, trade_date)

        # Load events
        event_result = await session.execute(
            select(Event).where(Event.run_id == run_id)
        )
        events_orm = event_result.scalars().all()
        events = []
        for evt in events_orm:
            stock_result = await session.execute(
                select(EventStock).where(EventStock.event_id == evt.id)
            )
            stocks = [s.stock_code for s in stock_result.scalars().all()]
            events.append({
                "event_id": evt.event_id,
                "event_type": evt.event_type,
                "title": evt.title,
                "summary": evt.summary,
                "content": evt.content,
                "source": evt.source,
                "publish_time": evt.publish_time,
                "narrative": evt.narrative or "",
                "event_strength": evt.event_strength or 0,
                "heat_score": evt.heat_score or 0,
                "strength": evt.strength or 0,
                "novelty": evt.novelty or 0,
                "scope": evt.scope or "individual",
                "keywords": evt.keywords or [],
                "sector_list": evt.sector_list or [],
                "sector_tags": evt.sector_tags or [],
                "symbol_list": stocks,
                "llm_prompt": evt.llm_prompt or "",
                "llm_response": evt.llm_response or "",
                "llm_model": evt.llm_model or "",
            })

        # Load sentiment scores
        sent_result = await session.execute(
            select(SentimentScore).where(SentimentScore.run_id == run_id)
        )
        sentiment_scores = []
        for s in sent_result.scalars().all():
            sentiment_scores.append({
                "target_id": s.target_id,
                "target_type": s.target_type,
                "symbol": s.symbol,
                "sentiment_score": s.sentiment_score or 0,
                "narrative_score": s.narrative_score or s.narrative_strength or 0,
                "hype_score": s.hype_score or 0,
                "consistency_score": s.consistency_score or s.consensus or 0,
                "risk_score": s.risk_score or 0,
                "confidence": s.confidence or 0.5,
                "heat": s.heat or 0,
                "consensus": s.consensus or s.consistency_score or 0.5,
                "diffusion_speed": s.diffusion_speed or 0,
                "narrative_strength": s.narrative_strength or s.narrative_score or 0,
                "keywords": s.keywords or [],
                "finbert_positive": s.finbert_positive or 0,
                "finbert_negative": s.finbert_negative or 0,
                "finbert_neutral": s.finbert_neutral or 0,
                "llm_prompt": s.llm_prompt or "",
                "llm_response": s.llm_response or "",
            })

        # Load capital flow records
        cf_result = await session.execute(
            select(CapitalFlowRecord).where(CapitalFlowRecord.run_id == run_id)
        )
        capital_flow_records = []
        for r in cf_result.scalars().all():
            capital_flow_records.append({
                "symbol": r.symbol, "stock_name": r.stock_name or "",
                "price": r.price or 0, "change_pct": r.change_pct or 0,
                "amount": r.amount or 0, "amount_wan": r.amount_wan or 0,
                "turnover_pct": r.turnover_pct or 0,
            })

        # Load sector flow records
        sf_result = await session.execute(
            select(SectorFlowRecord).where(SectorFlowRecord.run_id == run_id)
        )
        sector_flow_records = []
        for r in sf_result.scalars().all():
            sector_flow_records.append({
                "sector_code": r.sector_code or "", "sector_name": r.sector_name or "",
                "change_pct": r.change_pct or 0, "turnover_yi": r.turnover_yi or 0,
                "main_force_net": r.main_force_net or 0, "heat": r.heat or 0,
                "stock_count": r.stock_count or 0, "up_count": r.up_count or 0,
            })

        # Load capital flow summary from run metadata
        capital_flow_summary = (run.metadata_ or {}).get("capital_flow_summary", {}) if run.metadata_ else {}

        # Load leader candidates
        cand_result = await session.execute(
            select(LeaderCandidate).where(LeaderCandidate.run_id == run_id).order_by(LeaderCandidate.rank)
        )
        leader_candidates = []
        for c in cand_result.scalars().all():
            leader_candidates.append({
                "stock_code": c.stock_code, "stock_name": c.stock_name or "",
                "leader_score": c.leader_score or 0, "rank": c.rank or 0,
                "monster_potential": c.monster_potential or 0,
                "limit_up_prob": c.limit_up_prob or 0,
                "reasoning": c.reasoning or "", "sector": c.sector or "",
                "sentiment_sub": c.sentiment_sub or 0,
                "flow_sub": c.flow_sub or 0,
                "lhb_sub": c.lhb_sub or 0,
                "ml_sub": c.ml_sub or 0,
                "event_sub": c.event_sub or 0,
                "sector_tag_sub": c.sector_tag_sub or 0,
            })

        # Load risk flags
        rf_result = await session.execute(
            select(RiskFlag).where(RiskFlag.run_id == run_id)
        )
        risk_flags = []
        for f in rf_result.scalars().all():
            risk_flags.append({
                "stock_code": f.stock_code, "risk_type": f.risk_type,
                "severity": f.severity or 0, "description": f.description or "",
            })

        # Load dragon tiger records
        dt_result = await session.execute(
            select(DragonTigerRecord).where(DragonTigerRecord.run_id == run_id)
        )
        dragon_tiger_records = []
        for d in dt_result.scalars().all():
            dragon_tiger_records.append({
                "stock_code": d.stock_code, "stock_name": d.stock_name or "",
                "total_buy": d.total_buy or 0, "total_sell": d.total_sell or 0,
                "net_amount": d.net_amount or 0, "reason": d.reason or "",
            })

        # Load active stocks
        active_stocks = run.watchlist or []
        if not active_stocks:
            active_stocks = [c["stock_code"] for c in leader_candidates[:5]]

        # Detect placeholder runs (no real data at all)
        if len(events) == 0 and len(sentiment_scores) == 0 and len(sector_flow_records) == 0:
            logger.info("[/run] placeholder run detected for %s — treating as no data", trade_date)
            return None

        # Detect cross-month contamination: events with publish_time far after trade_date
        # (happens when batch backfill fetches current news for historical dates)
        if events and trade_date_obj:
            future_events = 0
            total_dated = 0
            for evt in events:
                pt = evt.get("publish_time", "")
                if pt:
                    total_dated += 1
                    if pt[:10] > trade_date[:10]:
                        future_events += 1
            if total_dated > 0 and future_events / total_dated > 0.5:
                logger.info("[/run] contaminated cache for %s — %d/%d events are future-dated, treating as no data",
                            trade_date, future_events, total_dated)
                return None

        logger.info("[/run] loaded from DB: %d events, %d candidates, %d sentiments",
                    len(events), len(leader_candidates), len(sentiment_scores))

        return {
            "events": events,
            "sentiment_scores": sentiment_scores,
            "capital_flow_records": capital_flow_records,
            "sector_flow_records": sector_flow_records,
            "capital_flow_summary": capital_flow_summary,
            "active_stocks": active_stocks,
            "dragon_tiger_records": dragon_tiger_records,
            "leader_candidates": leader_candidates,
            "risk_flags": risk_flags,
            "metadata": run.metadata_ or {},
        }


@app.post("/run", response_model=RunResponse)
async def run_pipeline(req: RunRequest):
    """Execute the full event → sentiment → candidates pipeline. DB-first cache."""
    raw_date = req.trade_date or get_trading_date_str()
    trade_date = get_last_trading_day(raw_date).isoformat() if raw_date else get_trading_date_str()
    logger.info("[/run] raw_date=%s → trade_date=%s top_n=%d", raw_date, trade_date, req.top_n)

    # 1. Try DB cache first (skip if force=True)
    if not req.force:
        cached = await _load_run_from_db(trade_date)
        if cached is not None:
            return RunResponse(
                run_id=cached["metadata"].get("started_at", trade_date),
                status="completed (cached)",
                events=cached["events"],
                sentiment_scores=cached["sentiment_scores"],
                capital_flow_records=cached["capital_flow_records"],
                sector_flow_records=cached["sector_flow_records"],
                capital_flow_summary=cached["capital_flow_summary"],
                active_stocks=cached["active_stocks"],
                dragon_tiger_records=cached["dragon_tiger_records"],
                leader_candidates=cached["leader_candidates"],
                risk_flags=cached["risk_flags"],
                metadata=cached["metadata"],
            )

    # 2. No cache — if not forced, return no_data (avoids running LLM + wrong news for historical dates)
    if not req.force:
        return RunResponse(
            status=f"no_data ({trade_date})",
            metadata={"message": f"{trade_date} 暂无分析数据，请先运行当日分析或勾选强制运行"},
        )

    # 3. Force run — execute full pipeline (calls LLM, external APIs)
    logger.info("[/run] force-running pipeline for %s", trade_date)
    graph = get_graph()
    initial_state = {
        "trade_date": trade_date,
        "top_n": req.top_n,
        "metadata": {"started_at": datetime.now().isoformat()},
    }

    from services.token_tracker import TokenUsageTracker
    TokenUsageTracker.instance().start_run(trade_date)

    result = await graph.ainvoke(initial_state)

    try:
        saved_run_id = await persist_run(trade_date, result)
        logger.info("[/run] persisted run_id=%s", saved_run_id)
    except Exception as exc:
        logger.error("[/run] persist failed: %s", exc)

    return RunResponse(
        run_id=result.get("metadata", {}).get("started_at", ""),
        status="completed",
        events=result.get("events", []),
        sentiment_scores=result.get("sentiment_scores", []),
        capital_flow_records=result.get("capital_flow_records", []),
        sector_flow_records=result.get("sector_flow_records", []),
        capital_flow_summary=result.get("capital_flow_summary", {}),
        active_stocks=result.get("active_stocks", []),
        dragon_tiger_records=result.get("dragon_tiger_records", []),
        leader_candidates=result.get("leader_candidates", []),
        risk_flags=result.get("risk_flags", []),
        metadata=result.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# History routes
# ---------------------------------------------------------------------------

@app.get("/history")
async def list_pipeline_history(days: int = 90):
    """List recent pipeline runs with summary stats."""
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=days)
    async with async_session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date >= cutoff
            ).order_by(PipelineRun.trade_date.desc())
        )
        runs = result.scalars().all()
        return [
            {
                "run_id": r.run_id,
                "trade_date": str(r.trade_date) if hasattr(r.trade_date, 'isoformat') else r.trade_date,
                "status": r.status,
                "event_count": r.event_count,
                "candidate_count": r.candidate_count,
                "top_score": r.top_score,
            }
            for r in runs
        ]


# ---------------------------------------------------------------------------
# Backtest routes
# ---------------------------------------------------------------------------

@app.get("/backtest/strategies")
async def list_strategies():
    """List all available trading strategies."""
    registry = get_registry()
    return [s.to_dict() for s in registry.list_all()]


@app.post("/backtest/strategies")
async def create_strategy(config: StrategyConfigRequest):
    """Create a new custom trading strategy."""
    registry = get_registry()
    try:
        s = TradingStrategy.from_dict(config.model_dump())
        registry.add(s)
        return {"status": "ok", "name": s.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"未知的规则类型: {e}")


@app.delete("/backtest/strategies/{name}")
async def delete_strategy(name: str):
    """Delete a custom strategy (system strategies cannot be deleted)."""
    registry = get_registry()
    try:
        registry.remove(name)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _load_prices(session, start_d, end_d) -> tuple[dict, dict]:
    """Load ALL price data for the date range.
    Returns (prices_by_date, close_by_date) — each is {date_str: {symbol: price}}
    Loads every symbol because engine needs today's prices to process yesterday's candidates.
    """
    from db.models import StockDailyBar
    from sqlalchemy import select, and_

    prices_by_date: dict[str, dict[str, float]] = {}
    close_by_date: dict[str, dict[str, float]] = {}

    rows = (await session.execute(
        select(StockDailyBar).where(
            and_(
                StockDailyBar.trade_date >= start_d,
                StockDailyBar.trade_date <= end_d,
            )
        ).order_by(StockDailyBar.trade_date.asc())
    )).scalars().all()

    for r in rows:
        ds = str(r.trade_date)
        price = r.open if r.open > 0 else r.close
        if price <= 0:
            continue  # skip zero/negative prices (data gap)
        if ds not in prices_by_date:
            prices_by_date[ds] = {}
            close_by_date[ds] = {}
        prices_by_date[ds][r.symbol] = price
        close_by_date[ds][r.symbol] = r.close if r.close > 0 else price

    total_symbols = len({s for d in prices_by_date for s in prices_by_date[d]})
    return prices_by_date, close_by_date


async def _load_bars(session, symbols: set[str], start_d, end_d) -> dict[str, list[dict]]:
    """Load OHLCV bars for each symbol, returning {symbol: [bar_dict, ...]}."""
    from db.models import StockDailyBar
    from sqlalchemy import select, and_

    if not symbols:
        return {}

    rows = (await session.execute(
        select(StockDailyBar).where(
            and_(
                StockDailyBar.trade_date >= start_d,
                StockDailyBar.trade_date <= end_d,
                StockDailyBar.symbol.in_(symbols),
            )
        ).order_by(StockDailyBar.trade_date.asc())
    )).scalars().all()

    result: dict[str, list[dict]] = {sym: [] for sym in symbols}
    for r in rows:
        result[r.symbol].append({
            "trade_date": str(r.trade_date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "turnover_pct": r.turnover_pct or 0,
        })
    return result


async def _load_index_bars(session, start_d, end_d) -> list[dict]:
    """Load 上证指数 bars."""
    from db.models import StockDailyBar
    from sqlalchemy import select

    rows = (await session.execute(
        select(StockDailyBar).where(
            StockDailyBar.trade_date >= start_d,
            StockDailyBar.trade_date <= end_d,
            StockDailyBar.symbol == "000001.SH",
        ).order_by(StockDailyBar.trade_date.asc())
    )).scalars().all()

    return [
        {"trade_date": str(r.trade_date),
         "open": r.open, "high": r.high, "low": r.low,
         "close": r.close, "volume": r.volume}
        for r in rows
    ]


async def _load_regime_fields(
    session, start_d, end_d
) -> dict[str, dict]:
    """Compute per-day regime fields: volatility, breadth, limit_up_count."""
    from statistics import stdev
    from db.models import StockDailyBar
    from sqlalchemy import select, func

    regime_by_date: dict[str, dict] = {}

    # ── 1. Load index bars ──
    idx_rows = (await session.execute(
        select(StockDailyBar).where(
            StockDailyBar.trade_date >= start_d,
            StockDailyBar.trade_date <= end_d,
            StockDailyBar.symbol == "000001.SH",
        ).order_by(StockDailyBar.trade_date.asc())
    )).scalars().all()

    idx_closes: list[tuple[str, float]] = []
    for r in idx_rows:
        idx_closes.append((str(r.trade_date), r.close if r.close > 0 else r.open))

    # ── 2. Volatility ──
    for i, (ds, close) in enumerate(idx_closes):
        lookback = idx_closes[max(0, i - 19):i + 1]
        if len(lookback) >= 5:
            returns = []
            for j in range(1, len(lookback)):
                if lookback[j - 1][1] > 0:
                    returns.append((lookback[j][1] - lookback[j - 1][1]) / lookback[j - 1][1])
            vol = stdev(returns) * (252 ** 0.5) if len(returns) >= 3 else 0.02
        else:
            vol = 0.02
        regime_by_date.setdefault(ds, {})["volatility"] = round(vol, 4)

    # ── 3. Breadth ──
    for i, (ds, close) in enumerate(idx_closes):
        lookback = idx_closes[max(0, i - 19):i + 1]
        if len(lookback) >= 5:
            ma20 = sum(c for _, c in lookback) / len(lookback)
            if ma20 > 0:
                ratio = close / ma20
                breadth = max(0.25, min(0.75, 0.5 + (ratio - 1) * 2))
            else:
                breadth = 0.5
            breadth = round(breadth, 4)
        else:
            breadth = 0.5
        regime_by_date.setdefault(ds, {})["breadth"] = breadth

    # ── 4. limit_up_count ──
    rows = (await session.execute(
        select(StockDailyBar.trade_date, func.count(StockDailyBar.symbol)).where(
            StockDailyBar.trade_date >= start_d,
            StockDailyBar.trade_date <= end_d,
            StockDailyBar.open > 0,
            StockDailyBar.close > 0,
            StockDailyBar.close > StockDailyBar.open,
            (StockDailyBar.close - StockDailyBar.open) / StockDailyBar.open >= 0.07,
        ).group_by(StockDailyBar.trade_date).order_by(StockDailyBar.trade_date.asc())
    )).all()
    for trade_date, cnt in rows:
        regime_by_date.setdefault(str(trade_date), {})["limit_up_count"] = cnt

    # ── 5. Fill defaults ──
    for fields in regime_by_date.values():
        fields.setdefault("volatility", 0.02)
        fields.setdefault("breadth", 0.5)
        fields.setdefault("limit_up_count", 0)

    return regime_by_date


@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    """Run a backtest using a specified strategy over a date range."""
    from datetime import date as date_type
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun, LeaderCandidate, StockDailyBar

    registry = get_registry()
    strategy_name = req.strategy_name  # V4 uses same exact strategy as standalone for identical results
    strategy = registry.get(strategy_name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    try:
        start_d = date_type.fromisoformat(req.start_date) if req.start_date else date_type.today()
        end_d = date_type.fromisoformat(req.end_date) if req.end_date else date_type.today()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    try:
        async with async_session_factory() as session:
            # Load price data from stock_daily_bars (needed for both modes)
            prices_by_date, close_by_date = await _load_prices(
                session, start_d, end_d
            )

            # Pre-load bars for V4 mode (needed for holding_scorer + index)
            bars_by_symbol_raw: dict[str, list[dict]] = {}
            idx_bars_raw: list[dict] = []
            if req.use_v4:
                from sqlalchemy import text as _sa_text
                raw_bars = await session.execute(
                    _sa_text(
                        "SELECT symbol, trade_date::text, open, high, low, close, volume "
                        "FROM stock_daily_bars_v2 "
                        "WHERE trade_date >= :s AND trade_date <= :e "
                        "  AND close > 0 AND open > 0 ORDER BY symbol, trade_date"
                    ), {"s": start_d, "e": end_d})
                for sym, td, o, h, l, c, v in raw_bars.fetchall():
                    if sym not in bars_by_symbol_raw:
                        bars_by_symbol_raw[sym] = []
                    bars_by_symbol_raw[sym].append({
                        "trade_date": td, "open": float(o or 0),
                        "high": float(h or 0), "low": float(l or 0),
                        "close": float(c or 0), "volume": float(v or 0),
                    })
                for sym in bars_by_symbol_raw:
                    bars_by_symbol_raw[sym].sort(key=lambda b: b["trade_date"])
                idx_bars_raw = bars_by_symbol_raw.get("000001.SH", [])
                logger.info("[V4] pre-loaded %d symbols bars", len(bars_by_symbol_raw))

            if req.use_scanner:
                # ── SCANNER MODE: full-market dragon_score scan ──
                from datetime import timedelta
                from services.backtest.market_scanner import scan_market

                # Load ALL bars with 120-day lookback via raw SQL (avoids ORM overhead)
                scan_start = start_d - timedelta(days=120)
                from sqlalchemy import text as sa_text
                raw_result = await session.execute(
                    sa_text(
                        "SELECT symbol, trade_date::text, open, high, low, close, volume "
                        "FROM stock_daily_bars_v2 "
                        "WHERE trade_date >= :s AND trade_date <= :e "
                        "ORDER BY symbol, trade_date"
                    ),
                    {"s": scan_start, "e": end_d}
                )
                bars_by_symbol: dict[str, list[dict]] = {}
                row_count = 0
                for row in raw_result.fetchall():
                    sym, td, o, h, l, c, v = row
                    if sym not in bars_by_symbol:
                        bars_by_symbol[sym] = []
                    bars_by_symbol[sym].append({
                        "trade_date": td,
                        "open": float(o or 0),
                        "high": float(h or 0),
                        "low": float(l or 0),
                        "close": float(c or 0),
                        "volume": float(v or 0),
                    })
                    row_count += 1
                logger.info("Scanner mode: loaded %d symbols, %d total bars",
                            len(bars_by_symbol), row_count)

                # Get trading days from prices data
                scan_dates = sorted(prices_by_date.keys())
                scan_dates = [d for d in scan_dates if start_d.isoformat() <= d <= end_d.isoformat()]

                # Build daily_runs from scanner
                daily_runs: list[dict] = []
                all_symbols: set[str] = set()
                # Index bars now included in v2 via merge

                # Pre-sort each symbol's bars (already sorted, but ensure)
                # and init progressive cursors to avoid O(n) re-scan per day
                cursors: dict[str, int] = {sym: 0 for sym in bars_by_symbol}

                for td in scan_dates:
                    if td not in prices_by_date:
                        continue

                    prices_today = prices_by_date.get(td, {})
                    day_bars = {}

                    # Only scan symbols that traded today (not all 5207)
                    for sym in prices_today:
                        sym_bars = bars_by_symbol.get(sym)
                        if not sym_bars:
                            continue

                        # Progressive cursor: advance only new bars <= td
                        c = cursors.get(sym, 0)
                        while c < len(sym_bars) and sym_bars[c]["trade_date"] <= td:
                            c += 1
                        cursors[sym] = c

                        if c >= 21:
                            day_bars[sym] = sym_bars[:c]

                    # Scan market for top candidates
                    candidates = scan_market(day_bars, td, prices_today,
                                             top_n=req.scanner_top_n)

                    for c in candidates:
                        all_symbols.add(c["stock_code"])

                    # Index bars: progressive cursor for index too
                    daily_runs.append({
                        "trade_date": td,
                        "leader_candidates": candidates,
                        "prices": prices_today,
                        "bars": day_bars,
                        "index_bars": [b for b in bars_by_symbol.get("000001.SH", []) if b["trade_date"] <= td],
                    })

                # Build prev_close
                sorted_dates = sorted(close_by_date.keys())
                prev_close_by_date: dict[str, dict[str, float]] = {}
                for i, ds in enumerate(sorted_dates):
                    prev_td = None
                    for j in range(i - 1, -1, -1):
                        if is_trading_day(sorted_dates[j]):
                            prev_td = sorted_dates[j]
                            break
                    prev_close_by_date[ds] = close_by_date.get(prev_td, {}) if prev_td else {}

                # Inject prev_close + prev_day_change into each daily run
                date_idx_map = {d: i for i, d in enumerate(sorted_dates)}
                for dr in daily_runs:
                    td = dr["trade_date"]
                    dr["prev_close"] = prev_close_by_date.get(td, {})
                    prev_td = None
                    for j in range(date_idx_map.get(td, 0) - 1, -1, -1):
                        if is_trading_day(sorted_dates[j]):
                            prev_td = sorted_dates[j]
                            break
                    if prev_td:
                        dr["prev_day_change"] = {
                            sym: ((prices_by_date.get(td, {}).get(sym, 0)
                                   - prices_by_date.get(prev_td, {}).get(sym, 0))
                                  / max(prices_by_date.get(prev_td, {}).get(sym, 0), 0.01))
                            for sym in all_symbols
                            if sym in prices_by_date.get(td, {})
                            and sym in prices_by_date.get(prev_td, {})
                        }
                    else:
                        dr["prev_day_change"] = {}

                logger.info("Scanner built %d daily_runs, avg %.0f candidates/day",
                            len(daily_runs),
                            sum(len(d["leader_candidates"]) for d in daily_runs) / max(len(daily_runs), 1))

            elif req.use_v4:
                # ── V4 MODE: delegate directly to backtest_v2.run_backtest logic ──
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from backtest_v2 import run_backtest as _v4_run
                from collections import defaultdict
                from shared.market_regime import clear_cache
                clear_cache()

                # run_backtest expects a year string, but we need custom range.
                # Replicate its core logic with our date range.
                from backtest_v2 import get_trading_dates, get_top_stocks
                from shared.market_regime import detect_regime

                trading_dates = await get_trading_dates(str(start_d), str(end_d))
                logger.info("[V4] %d trading days", len(trading_dates))

                # Build bars_by_symbol exactly like standalone
                _bars_sym: dict[str, list[dict]] = defaultdict(list)
                _prices: dict[str, dict[str, float]] = defaultdict(dict)
                _closes: dict[str, dict[str, float]] = defaultdict(dict)

                for sym, sym_bars in bars_by_symbol_raw.items():
                    for b in sym_bars:
                        td_s = b["trade_date"]
                        _bars_sym[sym].append(b)
                        _prices[td_s][sym] = b["open"]
                        _closes[td_s][sym] = b["close"]

                for sym in _bars_sym:
                    _bars_sym[sym].sort(key=lambda x: x["trade_date"])

                # Build prev_close (same as standalone)
                _sorted_dates = sorted(_closes.keys())
                _prev_close: dict[str, dict[str, float]] = {}
                for i, ds in enumerate(_sorted_dates):
                    prev_td = None
                    for j in range(i - 1, -1, -1):
                        from shared.utils.trading_day import is_trading_day as _itd
                        if _itd(_sorted_dates[j]):
                            prev_td = _sorted_dates[j]; break
                    _prev_close[ds] = _closes.get(prev_td, {}) if prev_td else {}

                _idx = _bars_sym.get("000001.SH", [])

                daily_runs = []
                for td in trading_dates:
                    regime = await detect_regime(td)
                    candidates = await get_top_stocks(td, regime, top_n=60)

                    _day_bars = {}
                    for sym, sym_bars in _bars_sym.items():
                        fb = [b for b in sym_bars if b["trade_date"] <= td]
                        if len(fb) >= 21:
                            _day_bars[sym] = fb
                    _idx_cur = [b for b in _idx if b["trade_date"] <= td]

                    daily_runs.append({
                        "trade_date": td,
                        "leader_candidates": candidates,
                        "prices": _prices.get(td, {}),
                        "prev_close": _prev_close.get(td, {}),
                        "bars": _day_bars,
                        "index_bars": _idx_cur,
                        "prev_day_change": {},
                        "sector_volume_pct": {}, "avg_volume_20": {},
                        "turnover_pct": {}, "today_volume": {},
                        "total_market_volume": 0, "sector_volume": {},
                        "breadth": 0.5, "volatility": 0.02,
                        "limit_up_count": 0, "sector_concentration": 0, "lianban_height": 0,
                        "score_sort_desc": True,
                    })
                logger.info("[V4] built %d daily_runs", len(daily_runs))

            else:
                # ── PIPELINE MODE: original 同花顺 hot-stock candidates ──
                result = await session.execute(
                    select(PipelineRun).where(
                        PipelineRun.trade_date >= start_d,
                        PipelineRun.trade_date <= end_d,
                    ).order_by(PipelineRun.trade_date.asc())
                )
                runs = result.scalars().all()

                daily_runs: list[dict] = []
                for run in runs:
                    cands_result = await session.execute(
                        select(LeaderCandidate).where(
                            LeaderCandidate.run_id == run.run_id
                        ).order_by(LeaderCandidate.rank)
                    )
                    cands = cands_result.scalars().all()
                    daily_runs.append({
                        "trade_date": str(run.trade_date),
                        "leader_candidates": [
                            {
                                "stock_code": c.stock_code,
                                "stock_name": c.stock_name,
                                "leader_score": c.leader_score,
                                "lhb_score": c.lhb_sub,
                                "flow_score": c.flow_sub,
                                "concept_score": c.sector_tag_sub,
                                "sector": c.sector or "",
                            }
                            for c in cands
                        ],
                    })

                # Build prev_close: for each day, use previous TRADING day's close
                sorted_dates = sorted(close_by_date.keys())
                prev_close_by_date: dict[str, dict[str, float]] = {}
                for i, ds in enumerate(sorted_dates):
                    prev_td = None
                    for j in range(i - 1, -1, -1):
                        if is_trading_day(sorted_dates[j]):
                            prev_td = sorted_dates[j]
                            break
                    if prev_td:
                        prev_close_by_date[ds] = close_by_date.get(prev_td, {})
                    else:
                        prev_close_by_date[ds] = {}

                # Collect all symbols that appear in leader_candidates (any date)
                all_symbols = set()
                for dr in daily_runs:
                    for c in dr["leader_candidates"]:
                        all_symbols.add(c["stock_code"])

                # Load OHLCV bars for holding scorer
                bars_by_symbol = await _load_bars(session, all_symbols, start_d, end_d)
                index_bars = await _load_index_bars(session, start_d, end_d)

            # Load market regime fields: volatility, breadth, limit_up_count
            regime_fields = await _load_regime_fields(session, start_d, end_d)
            logger.info("Regime fields loaded for %d dates", len(regime_fields))

            # Build date index for prev_day_change
            sorted_dates_sym = sorted(prices_by_date.keys())
            date_idx_map = {d: i for i, d in enumerate(sorted_dates_sym)}

            # Inject fields into each daily run (scanner mode already has these set)
            for dr in daily_runs:
                td = dr["trade_date"]

                if not req.use_scanner and not req.use_v4:
                    dr["prices"] = prices_by_date.get(td, {})
                    dr["prev_close"] = prev_close_by_date.get(td, {})

                    # Date-filter bars: only up to this trade_date (no look-ahead)
                    day_bars: dict[str, list[dict]] = {}
                    for sym, sym_bars in bars_by_symbol.items():
                        filtered = [b for b in sym_bars if b.get("trade_date", "") <= td]
                        if filtered:
                            day_bars[sym] = filtered
                    dr["bars"] = day_bars

                    dr["index_bars"] = [b for b in index_bars if b.get("trade_date", "") <= td]

                    # prev_day_change — use previous TRADING day for OneDaySpikeFilter
                    prev_td = None
                    for j in range(date_idx_map.get(td, 0) - 1, -1, -1):
                        if is_trading_day(sorted_dates_sym[j]):
                            prev_td = sorted_dates_sym[j]
                            break
                    if prev_td:
                        dr["prev_day_change"] = {
                            sym: ((prices_by_date.get(td, {}).get(sym, 0)
                                   - prices_by_date.get(prev_td, {}).get(sym, 0))
                                  / max(prices_by_date.get(prev_td, {}).get(sym, 0), 0.01))
                            for sym in all_symbols
                            if sym in prices_by_date.get(td, {})
                            and sym in prices_by_date.get(prev_td, {})
                        }
                    else:
                        dr["prev_day_change"] = {}

                # Context fields from real market data
                rf = regime_fields.get(td, {})
                dr["sector_volume_pct"] = {}
                dr["avg_volume_20"] = {}
                dr["turnover_pct"] = {}
                dr["today_volume"] = {}
                dr["sector_volume"] = {}
                dr["total_market_volume"] = 0
                dr["breadth"] = rf.get("breadth", 0.5)
                dr["volatility"] = rf.get("volatility", 0.02)
                dr["limit_up_count"] = rf.get("limit_up_count", 0)
                dr["sector_concentration"] = rf.get("sector_concentration", 0)
                dr["lianban_height"] = rf.get("lianban_height", 0)

        # Sanitize: replace None scores with 0 to prevent NoneType errors
        for dr in daily_runs:
            for c in dr.get("leader_candidates", []):
                if c.get("leader_score") is None:
                    c["leader_score"] = 0.0
                if c.get("dragon_score") is None:
                    c["dragon_score"] = 0.0

        # Check cache (skip if code changed)
        import hashlib, json as _json, os as _os
        _proj = Path(__file__).resolve().parent.parent.parent  # D:/K/dragon-engine
        _cache_dir = _proj / "data_lake" / "backtest_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _code_key = str(_os.path.getmtime(_proj / "services" / "backtest" / "engine.py")) + \
                    str(_os.path.getmtime(_proj / "services" / "backtest" / "strategies.py"))
        _req_hash = hashlib.md5(f"{req.strategy_name}{req.start_date}{req.end_date}{req.use_scanner}{req.use_v4}{_code_key}".encode()).hexdigest()[:12]
        _cache_file = _cache_dir / f"{_req_hash}.json"

        if _cache_file.exists():
            logger.info("[backtest] CACHE HIT %s", _req_hash)
            return _json.loads(_cache_file.read_text(encoding="utf-8"))

        engine = BacktestEngine(strategy)
        result = engine.run(daily_runs)

        # Phase 2: Reflection analysis
        reflection = reflect_analyze(result.trades)
        logger.info(
            "Reflection: %d trades, %.0f%% WR, %d flags",
            reflection.total_trades,
            reflection.win_rate * 100,
            len(reflection.flags),
        )

        response_data = {
            "strategy_name": result.strategy_name,
            "start_date": str(result.start_date),
            "end_date": str(result.end_date),
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "total_commission": result.total_commission,
            "total_stamp_duty": result.total_stamp_duty,
            "trades": [
                {
                    "stock_code": t.stock_code,
                    "stock_name": t.stock_name,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "cost": t.cost,
                    "proceeds": t.proceeds,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "entry_commission": t.entry_commission,
                    "exit_commission": t.exit_commission,
                    "stamp_duty": t.stamp_duty,
                    "net_pnl": t.net_pnl,
                    "entry_score": t.entry_score,
                    "exit_score": t.exit_score,
                    "exit_reason": t.exit_reason,
                    "holding_days": t.holding_days,
                    "cash_after_trade": round(t.cash_after_trade, 2),
                }
                for t in result.trades
            ],
            "daily_snapshots": [
                {"date": str(s.date), "equity": round(s.equity, 2), "cash": round(s.cash, 2), "regime": s.regime}
                for s in result.daily_snapshots
            ],
            "reflection": {
                "total_trades": reflection.total_trades,
                "win_rate": reflection.win_rate,
                "avg_pnl": reflection.avg_pnl,
                "total_pnl": reflection.total_pnl,
                "exit_reason_stats": reflection.exit_reason_stats,
                "holding_stats": reflection.holding_stats,
                "score_stats": reflection.score_stats,
                "flags": reflection.flags,
            },
        }
        # Save to cache
        _cache_file.write_text(_json.dumps(response_data, ensure_ascii=False), encoding="utf-8")
        logger.info("[backtest] cached to %s", _cache_file.name)
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Bar data backfill route
# ---------------------------------------------------------------------------

@app.post("/backfill/bars")
async def backfill_bars(days: int = 7, all_stocks: bool = False):
    """Backfill missing OHLCV bars via mootdx. Returns {symbols, rows, failed}."""
    import asyncio as _asyncio
    from datetime import date as _date, timedelta as _td
    from mootdx.quotes import Quotes
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import StockDailyBar
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    today = _date.today()
    target_dates = [(today - _td(days=i)).isoformat() for i in range(1, days + 1)]

    async with async_session_factory() as session:
        source = "stock_daily_bars_v2" if all_stocks else (
            "(SELECT DISTINCT lc.stock_code AS symbol FROM leader_candidates lc "
            "JOIN pipeline_runs pr ON lc.run_id = pr.run_id "
            "WHERE pr.trade_date >= CURRENT_DATE - INTERVAL '30 days')"
        )
        date_list = ", ".join(f"'{d}'::date" for d in target_dates)
        r = await session.execute(sa_text(
            f"SELECT s.symbol FROM {source} s "
            f"WHERE s.symbol NOT IN (SELECT DISTINCT symbol FROM stock_daily_bars_v2 "
            f"WHERE trade_date IN ({date_list}) AND open != close) ORDER BY s.symbol"
        ))
        symbols = [row[0] for row in r.fetchall()]

    if not symbols:
        return {"status": "ok", "symbols": 0, "rows": 0, "message": "all up to date"}

    logger.info("[backfill/bars] %d symbols missing bars for last %d days", len(symbols), days)

    min_date = _date.fromisoformat(target_dates[-1]) - _td(days=5)
    max_date = _date.fromisoformat(target_dates[0]) + _td(days=1)

    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
    def _fetch_one(sym):
        c = Quotes.factory(market="std", timeout=15)
        try:
            df = c.bars(symbol=sym, frequency=9, start=0, offset=100)
        except Exception as e:
            logger.warning("[backfill/bars] %s mootdx error: %s", sym, e)
            return []
        if df is None or df.empty:
            return []
        rows = []
        for _, row in df.iterrows():
            dt = row.get("datetime")
            if dt is None:
                continue
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
            td = dt.date() if hasattr(dt, 'date') else _date.fromisoformat(str(dt)[:10])
            if td < min_date or td > max_date:
                continue
            o = float(row.get("open", 0) or 0); h = float(row.get("high", 0) or 0)
            l = float(row.get("low", 0) or 0); c = float(row.get("close", 0) or 0)
            v = float(row.get("volume", 0) or 0); a = float(row.get("amount", 0) or 0)
            if all(x == 0 for x in [o, h, l, c]):
                continue
            rows.append(dict(symbol=sym, trade_date=td, open=o, high=h, low=l, close=c, volume=v, amount=a,
                             change_pct=0.0, turnover_pct=0.0))
        return rows

    inserted = 0; failed = 0; pending = []
    with _TPE(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        for fut in _ac(futures):
            sym = futures[fut]
            try:
                rows = fut.result(timeout=30)
            except Exception as e:
                logger.warning("[backfill/bars] %s thread error: %s", sym, e)
                rows = []
            if rows:
                pending.extend(rows)
            else:
                failed += 1

            if len(pending) >= 500:
                inserted += await _flush_bars(pending)
                pending = []

    if pending:
        inserted += await _flush_bars(pending)

    logger.info("[backfill/bars] done: %d symbols, %d rows, %d failed", len(symbols), inserted, failed)
    return {"status": "ok", "symbols": len(symbols), "rows": inserted, "failed": failed}


async def _flush_bars(pending: list) -> int:
    """Dedup and upsert pending bar rows. Returns number of unique rows inserted."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import StockDailyBar
    from db.connection import async_session_factory
    seen = set(); unique = []
    for r in pending:
        k = (r["symbol"], r["trade_date"])
        if k not in seen:
            seen.add(k); unique.append(r)
    stmt = pg_insert(StockDailyBar).values(unique)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "trade_date"],
        set_={"open": stmt.excluded.open, "high": stmt.excluded.high,
              "low": stmt.excluded.low, "close": stmt.excluded.close,
              "volume": stmt.excluded.volume, "amount": stmt.excluded.amount})
    async with async_session_factory() as s:
        async with s.begin():
            await s.execute(stmt)
    return len(unique)


# ---------------------------------------------------------------------------
# V4 candidate builder (5-factor scoring, no pipeline needed)
# ---------------------------------------------------------------------------

async def _build_v4_daily_runs(
    session, start_d, end_d, prices_by_date, close_by_date,
    bars_by_symbol: dict[str, list[dict]], idx_bars: list[dict],
    top_n: int = 60,
) -> list[dict]:
    """Build daily_runs using V4 5-factor scoring from stock_daily_bars_v2."""
    from sqlalchemy import text as sa_text
    from shared.market_regime import detect_regime, clear_cache
    from shared.filters import fetch_concept_leaders, find_oversold_bounce
    from shared.scorer import score_candidates

    clear_cache()

    # Get all trading dates in range
    from datetime import date as _dt_date
    r = await session.execute(
        sa_text(
            "SELECT DISTINCT trade_date FROM stock_daily_bars_v2 "
            "WHERE trade_date >= :s AND trade_date <= :e ORDER BY trade_date"
        ),
        {"s": start_d, "e": end_d},
    )
    trading_dates = [str(row[0]) for row in r.fetchall()]
    logger.info("[V4] %d trading days in range", len(trading_dates))

    daily_runs = []
    for td_str in trading_dates:
        td_obj = _dt_date.fromisoformat(td_str[:10])
        regime = await detect_regime(td_str)
        candidates: list[dict] = []
        symbols_seen: set[str] = set()

        # ---- Concept leaders (BULL / CHOPPY_UP) ----
        if regime in ("BULL", "CHOPPY_UP"):
            try:
                prev_r = await session.execute(
                    sa_text(
                        "SELECT trade_date FROM stock_daily_bars_v2 "
                        "WHERE trade_date < :td AND close > 0 "
                        "ORDER BY trade_date DESC LIMIT 1"
                    ),
                    {"td": td_obj},
                )
                prev_row = prev_r.fetchone()
                prev_td = str(prev_row[0]) if prev_row else td_str

                cl = await fetch_concept_leaders(prev_td, top_n=40)
                if cl:
                    cl_syms = [c["symbol"] for c in cl]
                    bars_r = await session.execute(
                        sa_text(
                            "SELECT symbol, close, change_pct, turnover_pct, amount, volume "
                            "FROM stock_daily_bars_v2 WHERE symbol = ANY(:syms) AND trade_date = :td"
                        ),
                        {"syms": cl_syms, "td": td_obj},
                    )
                    bars = {row[0]: row for row in bars_r.fetchall()}
                    for c in cl:
                        sym = c["symbol"]
                        if sym not in symbols_seen:
                            bar = bars.get(sym)
                            c["stock_code"] = sym
                            c["price"] = float(bar[1] or 0) if bar else 0
                            c["change_pct"] = float(bar[2] or 0) if bar else c.get("change_pct", 0)
                            c["turnover_pct"] = float(bar[3] or 0) if bar else 0
                            c["amount"] = float(bar[4] or 0) if bar else 0
                            c["volume"] = float(bar[5] or 0) if bar else 0
                            candidates.append(c)
                            symbols_seen.add(sym)
            except Exception as exc:
                logger.warning("[V4] concept leaders unavailable for %s: %s", td_str, exc)

        # ---- Oversold bounce (BEAR / CHOPPY_DOWN) ----
        if regime in ("BEAR", "CHOPPY_DOWN"):
            try:
                oversold = await find_oversold_bounce(td_str, top_n=top_n)
                for o in oversold:
                    if o["symbol"] not in symbols_seen:
                        o["stock_code"] = o["symbol"]
                        o["_source"] = "oversold_bounce"
                        o["leader_score"] = 0.0
                        candidates.append(o)
                        symbols_seen.add(o["symbol"])
            except Exception as exc:
                logger.warning("[V4] oversold unavailable for %s: %s", td_str, exc)

        # ---- Momentum stocks (all regimes, fills gaps) ----
        needed = top_n - len(candidates)
        if needed > 0:
            r2 = await session.execute(
                sa_text(
                    "SELECT symbol, close, change_pct, turnover_pct, amount, volume "
                    "FROM stock_daily_bars_v2 "
                    "WHERE trade_date = :td AND close > 0 AND volume > 0 "
                    "  AND close >= 3.0 AND amount >= 50000000 "
                    "ORDER BY amount DESC LIMIT :n"
                ),
                {"td": td_obj, "n": needed + len(candidates)},
            )
            for row in r2.fetchall():
                sym = row[0]
                if sym not in symbols_seen:
                    candidates.append({
                        "symbol": sym, "stock_code": sym, "stock_name": sym,
                        "price": float(row[1] or 0),
                        "change_pct": float(row[2] or 0),
                        "turnover_pct": float(row[3] or 0),
                        "amount": float(row[4] or 0),
                        "amount_wan": round(float(row[4] or 0) / 10000, 2),
                        "volume": float(row[5] or 0),
                        "leader_score": 0.0,
                    })
                    symbols_seen.add(sym)

        # ---- Score with 5 factors ----
        if candidates:
            candidates = score_candidates(candidates, trade_date=td_str, regime=regime)
            candidates = candidates[:top_n]
            for i, c in enumerate(candidates):
                c["rank"] = i + 1
                if "stock_code" not in c:
                    c["stock_code"] = c.get("symbol", "")

        # Build progressive day_bars (candidates only see bars up to td)
        day_bars = {}
        for sym in symbols_seen:
            sym_bars = bars_by_symbol.get(sym, [])
            filtered = [b for b in sym_bars if b["trade_date"] <= td_str]
            if len(filtered) >= 10:  # need some history for scorer
                day_bars[sym] = filtered
        idx_cur = [b for b in idx_bars if b["trade_date"] <= td_str]

        daily_runs.append({
            "trade_date": td_str,
            "leader_candidates": [
                {
                    "stock_code": c.get("stock_code") or c.get("symbol", ""),
                    "stock_name": c.get("stock_name", c.get("symbol", "")),
                    "leader_score": c.get("leader_score", 0),
                    "sector": c.get("concept_name", ""),
                }
                for c in candidates
            ],
            "prices": prices_by_date.get(td_str, {}),
            "prev_close": {},
            "bars": day_bars,
            "index_bars": idx_cur,
            "prev_day_change": {},
            "sector_volume_pct": {},
            "avg_volume_20": {},
            "turnover_pct": {},
            "today_volume": {},
            "sector_volume": {},
            "total_market_volume": 0,
            "breadth": 0.5,
            "volatility": 0.02,
            "limit_up_count": 0,
            "sector_concentration": 0,
            "lianban_height": 0,
            "score_sort_desc": True,
        })

        if len(daily_runs) % 20 == 0:
            logger.info("[V4] processed %d/%d days", len(daily_runs), len(trading_dates))

    logger.info("[V4] built %d daily_runs, avg %.0f candidates/day",
                len(daily_runs),
                sum(len(d["leader_candidates"]) for d in daily_runs) / max(len(daily_runs), 1))
    return daily_runs


# ---------------------------------------------------------------------------
# Token usage routes
# ---------------------------------------------------------------------------

@app.get("/token-usage")
async def get_token_usage(days: int = 30):
    """Get token usage summary for recent pipeline runs."""
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=days)
    async with async_session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date >= cutoff
            ).order_by(PipelineRun.trade_date.desc())
        )
        runs = result.scalars().all()
        return [
            {
                "run_id": r.run_id,
                "trade_date": str(r.trade_date) if hasattr(r.trade_date, 'isoformat') else r.trade_date,
                "token_usage": r.token_usage or {},
            }
            for r in runs
        ]


# ---------------------------------------------------------------------------
# Direct runner (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.graph_service.main:app", host="0.0.0.0", port=8000, reload=False)
