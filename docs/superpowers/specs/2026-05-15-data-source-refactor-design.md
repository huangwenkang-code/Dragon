# Data Source Refactor Design

**Date**: 2026-05-15 | **Status**: Approved | **Version**: 1.0

## Overview

Replace AKShare/Tushare data fetching with a unified `shared/data_sources/` layer using the DataSource registration pattern. Add mootdx (realtime), 腾讯财经 (fundamentals), 同花顺热点 (sector tags). Preserve AKShare as fallback for LHB data.

## Architecture

```
shared/data_sources/
├── interface.py       # DataSource ABC + DataQuery/DataResult
├── registry.py        # Source registry + factory
├── manager.py         # Cache (LRU+TTL) + rate limiting
├── sources/
│   ├── mootdx_source.py       # TCP realtime via mootdx lib
│   ├── tx_finance_source.py   # HTTP fundamentals via qt.gtimg.cn
│   └── ths_hot_source.py      # HTTP sector tags via 同花顺
└── constants.py       # QueryType enum, rate limits, TTLs
```

## Core Abstractions

- **DataSource(ABC)**: connect(), query(DataQuery) → DataResult, health_check(), default_rate_limit()
- **DataQuery**: query_type, symbols, date, extra
- **DataResult**: source, query_type, data: list[dict], raw, cached_at, error
- **QueryType enum**: REALTIME_QUOTE, FUNDAMENTALS, SECTOR_TAGS, HOT_RANKING, FUND_FLOW, LHB_BOARD, RESEARCH_REPORT, STOCK_NOTICE, BASIC_INFO

## Cache Strategy (in-memory LRU+TTL)

| Query Type | TTL | Reason |
|-----------|-----|--------|
| REALTIME_QUOTE | 3s | Real-time data, prevent duplicate pulls |
| SECTOR_TAGS | 300s | Heat changes slowly |
| FUNDAMENTALS | 3600s | PE/PB updates daily at most |
| LHB_BOARD | 86400s | Updated once per day after close |

## Migration Steps

1. Create `shared/data_sources/` base (interface + registry + manager)
2. Implement mootdx source
3. Implement 腾讯财经 source
4. Implement 同花顺热点 source
5. Rewire capital_flow node → mootdx
6. Rewire ingest_event node → ths_hot (add sector_tags to Event)
7. Rewire generate_candidates weights (add sector_tag_boost)
8. Add config switches (DATA_SOURCE_REALTIME, DATA_SOURCE_FUNDAMENTALS, DATA_SOURCE_SECTOR)
9. End-to-end verification (python -m services.graph_service.main)
