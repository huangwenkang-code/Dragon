"""SQLAlchemy ORM models — one per table, matching db/schema.sql v2.

All models use the async-capable DeclarativeBase from SQLAlchemy 2.0.
Table names map directly to pipeline node outputs.
"""

from datetime import datetime

from sqlalchemy import (BigInteger, JSON, Boolean, Column, Date, DateTime, Float, ForeignKey,
                        Integer, String, Text, Time, UniqueConstraint, text)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Helper: UTC now default for all created_at columns
# ---------------------------------------------------------------------------

def _utcnow():
    return datetime.utcnow()


# ===========================================================================
# 0. Stock basics
# ===========================================================================

class StockBasic(Base):
    __tablename__ = "stock_basics"

    stock_code      = Column(String(10), primary_key=True)
    stock_name      = Column(String(50), nullable=False)
    industry        = Column(String(50), default="")
    market_cap      = Column(Float, default=0)
    circulating_cap = Column(Float, default=0)
    pe              = Column(Float, default=0)
    pb              = Column(Float, default=0)
    list_date       = Column(String(10), default="")
    updated_at      = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ===========================================================================
# 1. Pipeline run
# ===========================================================================

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id          = Column(String(50), primary_key=True)
    trade_date      = Column(Date, nullable=False)
    status          = Column(String(20), default="completed")
    watchlist       = Column(ARRAY(Text), default=[])
    top_n           = Column(Integer, default=5)
    event_count     = Column(Integer, default=0)
    candidate_count = Column(Integer, default=0)
    top_score       = Column(Float, default=0)
    errors          = Column(ARRAY(Text), default=[])
    metadata_       = Column("metadata", JSON, default=dict)
    token_usage     = Column(JSON, default=dict)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    # relationships
    events                = relationship("Event", back_populates="run", cascade="all, delete-orphan")
    sentiment_scores      = relationship("SentimentScore", back_populates="run", cascade="all, delete-orphan")
    capital_flow_records  = relationship("CapitalFlowRecord", back_populates="run", cascade="all, delete-orphan")
    sector_flow_records   = relationship("SectorFlowRecord", back_populates="run", cascade="all, delete-orphan")
    dragon_tiger_records  = relationship("DragonTigerRecord", back_populates="run", cascade="all, delete-orphan")
    active_stocks         = relationship("ActiveStock", back_populates="run", cascade="all, delete-orphan")
    sector_concepts       = relationship("SectorConcept", back_populates="run", cascade="all, delete-orphan")
    stock_concept_tags    = relationship("StockConceptTag", back_populates="run", cascade="all, delete-orphan")
    leader_candidates     = relationship("LeaderCandidate", back_populates="run", cascade="all, delete-orphan")
    risk_flags            = relationship("RiskFlag", back_populates="run", cascade="all, delete-orphan")
    activated_memories    = relationship("ActivatedMemory", back_populates="run", cascade="all, delete-orphan")


# ===========================================================================
# 2. Events
# ===========================================================================

class Event(Base):
    __tablename__ = "events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    event_id        = Column(String(50), default="")
    event_type      = Column(String(20), default="")
    title           = Column(Text, nullable=False)
    summary         = Column(Text, default="")
    content         = Column(Text, default="")
    source          = Column(String(100), default="")
    publish_time    = Column(String(30), default="")
    narrative       = Column(Text, default="")
    event_strength  = Column(Float, default=0)
    heat_score      = Column(Float, default=0)
    strength        = Column(Float, default=0)
    novelty         = Column(Float, default=0)
    scope           = Column(String(20), default="individual")
    keywords        = Column(ARRAY(Text), default=[])
    sector_list     = Column(ARRAY(Text), default=[])
    sector_tags     = Column(ARRAY(Text), default=[])
    llm_prompt      = Column(Text, default="")
    llm_response    = Column(Text, default="")
    llm_model       = Column(String(50), default="")
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="events")
    stock_links = relationship("EventStock", back_populates="event", cascade="all, delete-orphan")


class EventStock(Base):
    __tablename__ = "event_stocks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    event_id    = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    stock_code  = Column(String(10), nullable=False)

    __table_args__ = (UniqueConstraint("event_id", "stock_code"),)

    event = relationship("Event", back_populates="stock_links")


# ===========================================================================
# 3. Sentiment scores
# ===========================================================================

class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    run_id            = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    target_id         = Column(String(50), default="")
    target_type       = Column(String(20), default="stock")
    symbol            = Column(String(10), default="")
    sentiment_score   = Column(Float, default=0)
    narrative_score   = Column(Float, default=0)
    hype_score        = Column(Float, default=0)
    consistency_score = Column(Float, default=0)
    risk_score        = Column(Float, default=0)
    confidence        = Column(Float, default=0.5)
    heat              = Column(Float, default=0)
    consensus         = Column(Float, default=0.5)
    diffusion_speed   = Column(Float, default=0)
    narrative_strength = Column(Float, default=0)
    keywords          = Column(ARRAY(Text), default=[])
    finbert_positive  = Column(Float, default=0)
    finbert_negative  = Column(Float, default=0)
    finbert_neutral   = Column(Float, default=0)
    llm_prompt        = Column(Text, default="")
    llm_response      = Column(Text, default="")
    created_at        = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="sentiment_scores")


# ===========================================================================
# 4. Capital flow records
# ===========================================================================

class CapitalFlowRecord(Base):
    __tablename__ = "capital_flow_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    symbol          = Column(String(10), nullable=False)
    stock_name      = Column(String(50), default="")
    price           = Column(Float, default=0)
    change_pct      = Column(Float, default=0)
    amount          = Column(Float, default=0)
    amount_wan      = Column(Float, default=0)
    turnover_pct    = Column(Float, default=0)
    main_force_net  = Column(Float, default=0)
    main_force_ratio = Column(Float, default=0)
    super_large_net = Column(Float, default=0)
    large_net       = Column(Float, default=0)
    mid_net         = Column(Float, default=0)
    small_net       = Column(Float, default=0)
    total_net       = Column(Float, default=0)
    northbound_net  = Column(Float, default=0)
    flow_ratio      = Column(Float, default=0)
    sector_flow     = Column(Float, default=0)
    flow_score      = Column(Float, default=0)
    pe              = Column(Float, default=0)
    pb              = Column(Float, default=0)
    market_cap      = Column(Float, default=0)
    data_source     = Column(String(20), default="")
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="capital_flow_records")


# ===========================================================================
# 5. Sector flow records (同花顺90行业板块)
# ===========================================================================

class SectorFlowRecord(Base):
    __tablename__ = "sector_flow_records"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    sector_code         = Column(String(20), default="")
    sector_name         = Column(String(100), nullable=False)
    change_pct          = Column(Float, default=0)
    turnover_yi         = Column(Float, default=0)
    main_force_net      = Column(Float, default=0)
    main_force_ratio    = Column(Float, default=0)
    super_large_net     = Column(Float, default=0)
    large_net           = Column(Float, default=0)
    heat                = Column(Integer, default=0)
    stock_count         = Column(Integer, default=0)
    up_count            = Column(Integer, default=0)
    down_count          = Column(Integer, default=0)
    leading_stock       = Column(String(10), default="")
    leading_stock_name  = Column(String(50), default="")
    leading_stock_change = Column(Float, default=0)
    created_at          = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="sector_flow_records")


# ===========================================================================
# 6. Dragon tiger records
# ===========================================================================

class DragonTigerRecord(Base):
    __tablename__ = "dragon_tiger_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    stock_code      = Column(String(10), nullable=False)
    stock_name      = Column(String(50), default="")
    trade_date      = Column(Date, nullable=False)
    reason          = Column(String(200), default="")
    buy_seats       = Column(JSON, default=list)
    sell_seats      = Column(JSON, default=list)
    total_buy       = Column(Float, default=0)
    total_sell      = Column(Float, default=0)
    net_amount      = Column(Float, default=0)
    famous_traders  = Column(ARRAY(Text), default=[])
    trader_signal   = Column(String(20), default="")
    lhb_score       = Column(Float, default=0)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="dragon_tiger_records")


# ===========================================================================
# 7. Active stocks (merge node output)
# ===========================================================================

class ActiveStock(Base):
    __tablename__ = "active_stocks"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    symbol          = Column(String(10), nullable=False)
    stock_name      = Column(String(50), default="")
    rank            = Column(Integer, default=0)
    active_score    = Column(Float, default=0)
    flow_score      = Column(Float, default=0)
    concept_score   = Column(Float, default=0)
    lhb_score       = Column(Float, default=0)
    main_force_net  = Column(Float, default=0)
    ddejingliang    = Column(Float, default=0)
    super_large_net = Column(Float, default=0)
    large_net       = Column(Float, default=0)
    mid_net         = Column(Float, default=0)
    small_net       = Column(Float, default=0)
    amount_wan      = Column(Float, default=0)
    change_pct      = Column(Float, default=0)
    pe              = Column(Float, default=0)
    pb              = Column(Float, default=0)
    market_cap      = Column(Float, default=0)
    data_source     = Column(String(20), default="")
    reasons         = Column(Text, default="")
    matched_concepts = Column(ARRAY(Text), default=[])
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="active_stocks")


# ===========================================================================
# 8. Sector concepts & stock-concept tags
# ===========================================================================

class SectorConcept(Base):
    __tablename__ = "sector_concepts"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    concept_name        = Column(String(100), nullable=False)
    concept_id          = Column(String(20), default="")
    leader_stock        = Column(String(10), default="")
    leader_stock_name   = Column(String(50), default="")
    leader_stock_change = Column(Float, default=0)
    change_pct          = Column(Float, default=0)
    heat                = Column(Integer, default=0)
    stock_count         = Column(Integer, default=0)
    snapshot_date       = Column(Date, nullable=False)
    created_at          = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (UniqueConstraint("concept_name", "snapshot_date"),)

    run = relationship("PipelineRun", back_populates="sector_concepts")


class StockConceptTag(Base):
    __tablename__ = "stock_concept_tags"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_id        = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    stock_code    = Column(String(10), nullable=False)
    concept_name  = Column(String(100), nullable=False)
    is_leader     = Column(Boolean, default=False)
    snapshot_date = Column(Date, nullable=False)

    __table_args__ = (UniqueConstraint("stock_code", "concept_name", "snapshot_date"),)

    run = relationship("PipelineRun", back_populates="stock_concept_tags")


# ===========================================================================
# 9. Leader candidates
# ===========================================================================

class LeaderCandidate(Base):
    __tablename__ = "leader_candidates"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    run_id            = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    stock_code        = Column(String(10), nullable=False)
    stock_name        = Column(String(50), default="")
    trade_date        = Column(Date, nullable=False)
    rank              = Column(Integer, default=0)
    leader_score      = Column(Float, default=0)
    monster_potential = Column(Float, default=0)
    limit_up_prob     = Column(Float, default=0)
    reasoning         = Column(Text, default="")
    sector            = Column(String(50), default="")
    sentiment_sub     = Column(Float, default=0)
    flow_sub          = Column(Float, default=0)
    lhb_sub           = Column(Float, default=0)
    ml_sub            = Column(Float, default=0)
    event_sub         = Column(Float, default=0)
    sector_tag_sub    = Column(Float, default=0)
    monster_reference = Column(JSON, default=None)  # {top_matches: [...], summary: "..."}
    created_at        = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="leader_candidates")


# ===========================================================================
# 10. Risk flags
# ===========================================================================

class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    stock_code  = Column(String(10), nullable=False)
    risk_type   = Column(String(30), default="")
    severity    = Column(Float, default=0)
    description = Column(Text, default="")
    created_at  = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="risk_flags")


# ===========================================================================
# 11. Activated memories (ChromaDB historical matches)
# ===========================================================================

class ActivatedMemory(Base):
    __tablename__ = "activated_memories"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(String(50), ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    current_event_title = Column(Text, default="")
    historical_summary  = Column(Text, default="")
    similarity          = Column(Float, default=0)
    lifecycle_stage     = Column(String(20), default="")
    memory_created_at   = Column(String(30), default="")
    created_at          = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="activated_memories")


# ===========================================================================
# 12. Monster stock data lake
# ===========================================================================

class MonsterStock(Base):
    __tablename__ = "monster_stock"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    stock_code        = Column(String(10), nullable=False)
    stock_name        = Column(String(30), nullable=False)
    primary_type      = Column(String(30), nullable=False)
    secondary_type    = Column(String(30), nullable=False)
    tags              = Column(ARRAY(Text), default=[])
    sector            = Column(String(50), default="")
    start_date        = Column(Date, nullable=False)
    end_date          = Column(Date, nullable=False)
    trading_days      = Column(Integer, default=0)
    start_price       = Column(Float, default=0)
    peak_price        = Column(Float, default=0)
    max_gain_pct      = Column(Float, default=0)
    market_cap_start  = Column(Float, default=0)
    market_cap_peak   = Column(Float, default=0)
    daily_turnover_avg_pre   = Column(Float, default=0)
    daily_turnover_avg_surge = Column(Float, default=0)
    limit_up_count           = Column(Integer, default=0)
    consecutive_boards_max   = Column(Integer, default=0)
    drawdown_pct      = Column(Float, default=0)
    key_traders       = Column(ARRAY(Text), default=[])
    similar_cases     = Column(ARRAY(Text), default=[])
    markdown_path     = Column(String(500), default="")
    created_at        = Column(DateTime(timezone=True), default=_utcnow)
    updated_at        = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class MonsterDailyBar(Base):
    __tablename__ = "monster_daily_bar"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    stock_code      = Column(String(10), nullable=False)
    trade_date      = Column(Date, nullable=False)
    open            = Column(Float, nullable=False)
    high            = Column(Float, nullable=False)
    low             = Column(Float, nullable=False)
    close           = Column(Float, nullable=False)
    volume          = Column(Integer, default=0)
    amount          = Column(Float, default=0)
    turnover_pct    = Column(Float, default=0)
    change_pct      = Column(Float, default=0)
    is_limit_up     = Column(Boolean, default=False)
    is_limit_down   = Column(Boolean, default=False)
    ma5             = Column(Float, nullable=True)
    ma10            = Column(Float, nullable=True)
    ma20            = Column(Float, nullable=True)
    volume_ratio    = Column(Float, nullable=True)
    phase           = Column(String(20), default="")


class MonsterMinuteBar(Base):
    __tablename__ = "monster_minute_bar"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    stock_code      = Column(String(10), nullable=False)
    trade_date      = Column(Date, nullable=False)
    bar_time        = Column(Time, nullable=False)
    open            = Column(Float, nullable=False)
    high            = Column(Float, nullable=False)
    low             = Column(Float, nullable=False)
    close           = Column(Float, nullable=False)
    volume          = Column(Integer, default=0)
    amount          = Column(Float, default=0)
    phase           = Column(String(20), default="")


# ===========================================================================
# 13. Backtest — strategies, runs, trades
# ===========================================================================

class BacktestStrategy(Base):
    __tablename__ = "backtest_strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, default="")
    config_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float, default=100000)
    final_equity = Column(Float, default=0)
    total_return_pct = Column(Float, default=0)
    max_drawdown_pct = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    win_rate = Column(Float, default=0)
    total_trades = Column(Integer, default=0)
    daily_snapshots = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(50), default="")
    entry_date = Column(Date, nullable=False)
    exit_date = Column(Date, nullable=False)
    entry_price = Column(Float, default=0)
    exit_price = Column(Float, default=0)
    shares = Column(Integer, default=0)
    cost = Column(Float, default=0)
    proceeds = Column(Float, default=0)
    pnl = Column(Float, default=0)
    pnl_pct = Column(Float, default=0)
    exit_reason = Column(String(200), default="")
    holding_days = Column(Integer, default=0)


# ===========================================================================
# 16. Stock daily bars — historical OHLCV for backtest
# ===========================================================================

class StockDailyBar(Base):
    __tablename__ = "stock_daily_bars_v2"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(10), nullable=False)
    trade_date  = Column(Date, nullable=False)
    open        = Column(Float, default=0)
    high        = Column(Float, default=0)
    low         = Column(Float, default=0)
    close       = Column(Float, default=0)
    volume      = Column(BigInteger, default=0)
    amount      = Column(Float, default=0)
    change_pct  = Column(Float, default=0)
    turnover_pct = Column(Float, default=0)
    created_at  = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (UniqueConstraint("symbol", "trade_date"),)
