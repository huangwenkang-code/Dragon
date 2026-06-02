"""AgentState — the universal state schema flowing through the LangGraph."""

from datetime import datetime
from typing import Annotated, Optional

# Python 3.11: TypedDict is in typing (not typing_extensions)
try:
    from typing import TypedDict  # py3.11+
except ImportError:
    from typing_extensions import TypedDict  # py<3.11

from pydantic import BaseModel, Field

# add_messages is a LangGraph built-in message reducer.
# It is only needed when AgentState is used inside a StateGraph.
# We guard the import so schemas remain usable without langgraph installed.
try:
    from langgraph.graph.message import add_messages
except ImportError:
    add_messages = None  # type: ignore[assignment]
    # Provide a no-op fallback for standalone schema usage.
    # When langgraph is installed, the real reducer will be used.


# ---------------------------------------------------------------------------
# Entity schemas
# ---------------------------------------------------------------------------

class Event(BaseModel):
    """A market-driving event extracted from news/policy/social media."""

    id: str = ""
    event_id: str = ""
    event_type: str = Field(description="政策/产业/公告/突发/题材")
    title: str
    content: str = ""
    summary: str = ""
    source: str = ""
    publish_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    symbol_list: list[str] = Field(default_factory=list)
    sector_list: list[str] = Field(default_factory=list)
    narrative: str = ""
    event_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="事件强度 0-1")
    heat_score: float = Field(default=0.0, ge=0.0, le=1.0, description="热度评分 0-1")
    keywords: list[str] = Field(default_factory=list)
    strength: float = Field(default=0.0, ge=0.0, le=1.0, description="事件强度")
    novelty: float = Field(default=0.0, ge=0.0, le=1.0, description="新颖性")
    scope: str = Field(default="individual", description="影响范围: individual/sector/market")
    sector_tags: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_prompt: str = ""
    llm_response: str = ""
    llm_model: str = ""


class SentimentScore(BaseModel):
    """Multi-dimensional sentiment for a stock/sector."""

    target_id: str
    target_type: str = Field(default="stock", description="stock / sector / event")
    symbol: str = ""
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0, description="情绪评分 -1到1")
    narrative_score: float = Field(default=0.0, ge=0.0, le=1.0, description="叙事强度 0-1")
    hype_score: float = Field(default=0.0, ge=0.0, le=1.0, description="炒作热度 0-1")
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="一致性 0-1")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="风险评分 0-1")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度 0-1")
    heat: float = Field(default=0.0, ge=0.0, le=1.0, description="舆情热度")
    consensus: float = Field(default=0.5, ge=0.0, le=1.0, description="一致性")
    diffusion_speed: float = Field(default=0.0, ge=0.0, le=1.0, description="扩散速度")
    narrative_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="叙事强度")
    keywords: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    finbert_positive: float = Field(default=0.0, ge=0.0, le=1.0)
    finbert_negative: float = Field(default=0.0, ge=0.0, le=1.0)
    finbert_neutral: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_prompt: str = ""
    llm_response: str = ""


class LeaderCandidate(BaseModel):
    """A stock identified as a potential leader / monster stock."""

    stock_code: str
    stock_name: str = ""
    leader_score: float = Field(default=0.0, ge=0.0, le=1.0, description="龙头概率")
    monster_potential: float = Field(default=0.0, ge=0.0, le=1.0, description="妖股潜力")
    limit_up_prob: float = Field(default=0.0, ge=0.0, le=1.0, description="涨停概率")
    reasoning: str = ""
    sector: str = ""
    rank: int = 0


class RiskFlag(BaseModel):
    """Risk signal detected for a candidate."""

    stock_code: str
    risk_type: str = Field(description="炸板/假突破/退潮/高开低走/接力失败")
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class CapitalFlowRecord(BaseModel):
    """Capital flow data for a single stock."""

    symbol: str
    stock_name: str = ""
    main_force_net: float = Field(default=0.0, description="主力净流入(万元)")
    northbound_net: float = Field(default=0.0, description="北向净流入(万元)")
    total_net: float = Field(default=0.0, description="总净流入(万元)")
    flow_ratio: float = Field(default=0.0, description="净流入占比")
    sector_flow: float = Field(default=0.0, description="板块资金流向")
    flow_score: float = Field(default=0.0, ge=0.0, le=1.0, description="资金流入评分")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DragonTigerRecord(BaseModel):
    """Dragon-tiger board record for a stock."""

    stock_code: str
    stock_name: str = ""
    trade_date: str = ""
    reason: str = Field(default="", description="上榜原因")
    buy_seats: list[dict] = Field(default_factory=list, description="买入席位")
    sell_seats: list[dict] = Field(default_factory=list, description="卖出席位")
    total_buy: float = Field(default=0.0, description="总买入(万元)")
    total_sell: float = Field(default=0.0, description="总卖出(万元)")
    net_amount: float = Field(default=0.0, description="净买入(万元)")
    famous_traders: list[str] = Field(default_factory=list, description="识别到的知名游资")
    trader_signal: str = Field(default="", description="游资信号: 合力做多/分歧/出货")
    lhb_score: float = Field(default=0.0, ge=0.0, le=1.0, description="龙虎榜信号强度")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# AgentState (TypedDict for LangGraph)
# ---------------------------------------------------------------------------

# Resolve the messages reducer (supports optional langgraph install)
_MESSAGES_TYPE: type = (
    Annotated[list, add_messages]
    if add_messages is not None
    else list  # fallback when langgraph is unavailable
)


class AgentState(TypedDict, total=False):
    """The unified state that flows through every node in the graph."""

    # Message stack (LangGraph built-in reducer when available)
    messages: _MESSAGES_TYPE  # type: ignore[valid-type]

    # Input
    trade_date: str
    watchlist: list[str]  # stock codes to scan

    # Events
    events: list[dict]  # List[Event] serialised as dict for TypedDict
    event_count: int

    # Sentiment
    sentiment_scores: list[dict]  # List[SentimentScore]
    sentiment_summary: str

    # Capital flow
    capital_flow_records: list[dict]  # List[CapitalFlowRecord]

    # Dragon-tiger board
    dragon_tiger_records: list[dict]  # List[DragonTigerRecord]

    # Leader candidates
    leader_candidates: list[dict]  # List[LeaderCandidate]
    top_n: int

    # Risk
    risk_flags: list[dict]  # List[RiskFlag]
    risk_blocked: list[str]  # stock codes blocked by risk

    # Historical event memory
    activated_memories: list[dict]  # historically similar active events

    # Sector tags (from 同花顺热点)
    sector_tags: list[dict]  # per-symbol concept/sector tag mappings

    # Raw 同花顺热点 items (for merge node enrichment with ddejingliang)
    ths_hot_items: list[dict]

    # Sector flow (from EastMoney 板块资金流)
    sector_flow_records: list[dict]  # sector-level capital flow

    # Capital flow summary
    capital_flow_summary: dict  # total_main_inflow, top_sectors, etc.

    # Active stocks (merged candidate pool from capital_flow + concepts + LHB)
    active_stocks: list[dict]

    # Metadata
    metadata: dict  # pipeline run id, timing, etc.
