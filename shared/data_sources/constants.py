"""Data source constants — query types, rate limits, cache TTLs."""
from enum import Enum


class QueryType(str, Enum):
    REALTIME_QUOTE = "realtime_quote"
    REALTIME_TICK = "realtime_tick"
    HISTORY_KLINE = "history_kline"
    FUNDAMENTALS = "fundamentals"
    SECTOR_TAGS = "sector_tags"
    HOT_RANKING = "hot_ranking"
    SECTOR_MEMBERS = "sector_members"
    FUND_FLOW = "fund_flow"
    LHB_BOARD = "lhb_board"
    RESEARCH_REPORT = "research_report"
    STOCK_NOTICE = "stock_notice"
    BASIC_INFO = "basic_info"


CACHE_TTL = {
    QueryType.REALTIME_QUOTE: 3,
    QueryType.REALTIME_TICK: 3,
    QueryType.HISTORY_KLINE: 1800,
    QueryType.FUNDAMENTALS: 3600,
    QueryType.SECTOR_TAGS: 300,
    QueryType.HOT_RANKING: 300,
    QueryType.SECTOR_MEMBERS: 300,
    QueryType.FUND_FLOW: 10,
    QueryType.LHB_BOARD: 86400,
    QueryType.RESEARCH_REPORT: 3600,
    QueryType.STOCK_NOTICE: 3600,
    QueryType.BASIC_INFO: 86400,
}

RATE_LIMITS = {
    "mootdx": 10,
    "tx_finance": 50,
    "ths_hot": 30,
    "eastmoney_notice": 30,
    "eastmoney_report": 30,
}
