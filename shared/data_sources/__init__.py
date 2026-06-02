"""Unified data source layer for dragon-engine.

Usage:
    from shared.data_sources import get_source, DataQuery, QueryType
    source = get_source("mootdx")
    result = await source.query(DataQuery(QueryType.REALTIME_QUOTE, symbols=["000001"]))
"""

from shared.data_sources.constants import CACHE_TTL, RATE_LIMITS, QueryType
from shared.data_sources.interface import DataQuery, DataResult, DataSource, RateLimit
from shared.data_sources.manager import DataSourceManager, get_manager
from shared.data_sources.registry import (
    clear,
    get,
    get as get_source,
    is_registered,
    list_all,
    register,
    unregister,
)
