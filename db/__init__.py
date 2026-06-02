"""dragon-engine database layer.

Usage:
    from db import init_db, get_session

    # Startup
    await init_db()

    # Query
    async for session in get_session():
        result = await session.execute(...)
"""
from db.connection import (async_session_factory, close_db, engine, get_session,
                           init_db)
from db.models import (ActiveStock, ActivatedMemory, Base, CapitalFlowRecord,
                       DragonTigerRecord, Event, EventStock, LeaderCandidate,
                       MonsterDailyBar, MonsterMinuteBar, MonsterStock,
                       PipelineRun, RiskFlag, SectorConcept, SectorFlowRecord,
                       SentimentScore, StockBasic, StockConceptTag)
from db.persist import persist_run

__all__ = [
    "Base",
    "PipelineRun",
    "Event",
    "EventStock",
    "SentimentScore",
    "CapitalFlowRecord",
    "SectorFlowRecord",
    "DragonTigerRecord",
    "ActiveStock",
    "SectorConcept",
    "StockConceptTag",
    "LeaderCandidate",
    "RiskFlag",
    "ActivatedMemory",
    "StockBasic",
    "MonsterStock",
    "MonsterDailyBar",
    "MonsterMinuteBar",
    "init_db",
    "close_db",
    "get_session",
    "persist_run",
    "async_session_factory",
    "engine",
]
