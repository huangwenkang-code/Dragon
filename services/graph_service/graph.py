"""Build and compile the dragon-engine StateGraph — capital-driven + LLM cognitive pipeline.

Pipeline topology:

                START
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   capital_flow  sector   dragon_tiger
   (auto-discover _flow    _board
    + fund flow)   │          │
       │           │          │
       └───────────┼──────────┘
                   ▼
        merge_active_stocks
                   │
                   ▼
     find_news_double_layer   (LLM: DeepSeek event extraction)
                   │
                   ▼
         analyze_sentiment    (keyword + LLM summary enrichment)
                   │
                   ▼
        generate_candidates   (core 3-layer: flow 40% + concept 35% + LHB 25%)
                   │
                   ▼
                  END

capital_flow auto-discovers active stocks from tx_finance + EastMoney fund flow.
sector_flow fetches ths_hot concept data + industry sector rankings.
dragon_tiger_board fetches all LHB data independently.
merge_active_stocks enriches with THS dde + EastMoney fund flow breakdown.
find_news_double_layer searches EastMoney news by concept → LLM extracts structured events.
analyze_sentiment computes keyword-based sentiment + LLM summary enrichment.
generate_candidates scores from 3 layers: flow 40% + concept 35% + LHB 25%.
"""

from langgraph.graph import END, START, StateGraph

from shared.schemas.agent_state import AgentState
from services.graph_service.nodes import (
    analyze_sentiment,
    capital_flow,
    dragon_tiger_board,
    generate_candidates,
    merge_active_stocks,
    find_news_double_layer,
)


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # -- register nodes --
    workflow.add_node("capital_flow", capital_flow)
    workflow.add_node("sector_flow", _sector_flow_node)
    workflow.add_node("dragon_tiger_board", dragon_tiger_board)
    workflow.add_node("merge_active_stocks", merge_active_stocks)
    workflow.add_node("find_news_double_layer", find_news_double_layer)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("generate_candidates", generate_candidates)

    # -- edges --
    # Phase 1: parallel data discovery
    workflow.add_edge(START, "capital_flow")
    workflow.add_edge(START, "sector_flow")
    workflow.add_edge(START, "dragon_tiger_board")

    # Phase 2: merge signals
    workflow.add_edge("capital_flow", "merge_active_stocks")
    workflow.add_edge("sector_flow", "merge_active_stocks")
    workflow.add_edge("dragon_tiger_board", "merge_active_stocks")

    # Phase 3: final scoring (LLM nodes find_news_double_layer + analyze_sentiment
    # bypassed — events don't affect leader_score, saves ¥0.05/day)
    workflow.add_edge("merge_active_stocks", "generate_candidates")
    workflow.add_edge("generate_candidates", END)

    return workflow.compile()


# -- sector_flow node (同花顺热点 + 行业板块排名, degraded gracefully) --
async def _sector_flow_node(state: AgentState) -> dict:
    """Fetch hot strong stocks + real industry sector flow data.

    Data sources:
      1. 同花顺热点 (zx.10jqka.com.cn) — hot stocks + concept tags + ddejingliang
      2. akshare stock_board_industry_summary_ths — 90 industry sectors with
         real net inflow (亿), turnover (亿), leader stock, up/down counts

    Parses reason tags into concept keywords for merge node enrichment.
    Returns real sector_flow_records (not computed from tags).
    """
    from shared.utils.logging import get_logger
    import time as _time

    logger = get_logger(__name__)

    trade_date = state.get("trade_date", "")
    if not trade_date:
        from shared.utils.trading_day import get_trading_date_str
        trade_date = get_trading_date_str()

    # ------------------------------------------------------------------
    # 1. 同花顺热点 — hot stocks + concept tags
    # ------------------------------------------------------------------
    url = (
        f"http://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{trade_date}/"
        f"orderby/date/orderway/desc/charset/GBK/"
    )
    _t0 = _time.time()
    logger.info("[sector_flow] fetching hot strong stocks for %s", trade_date)

    sector_tags: list[dict] = []
    discovered_symbols: list[str] = []
    ths_hot_items: list[dict] = []

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36"
                ),
            })

        if resp.status_code == 200:
            data = resp.json()
            elapsed = _time.time() - _t0

            if data.get("errocode") == 0:
                items = data.get("data", [])
                logger.info("[sector_flow] got %d strong stocks in %.1fs", len(items), elapsed)
                ths_hot_items = items

                for item in items:
                    code = str(item.get("code", "")).strip()
                    name = str(item.get("name", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                    change_pct = float(item.get("zhangfu", 0) or 0)

                    if not code:
                        continue
                    discovered_symbols.append(code)

                    if reason:
                        tags = [t.strip() for t in reason.replace("，", "+").split("+") if t.strip()]
                    else:
                        tags = []

                    for tag in tags:
                        sector_tags.append({
                            "concept_name": tag,
                            "concept_id": "",
                            "leader_stock": code,
                            "leader_stock_name": name,
                            "leader_stock_change": round(change_pct, 2),
                            "change_pct": 0,
                            "heat": 0,
                            "stock_count": 0,
                        })

                logger.info("[sector_flow] %d symbols → %d concept tags (top: %s)",
                            len(discovered_symbols), len(sector_tags),
                            [t["concept_name"] for t in sector_tags[:12]])
            else:
                logger.warning("[sector_flow] API error: %s", data.get("erromsg", ""))
        else:
            logger.warning("[sector_flow] zx.10jqka returned HTTP %d", resp.status_code)
    except Exception as exc:
        logger.warning("[sector_flow] 同花顺热点 failed: %s", exc)

    # ------------------------------------------------------------------
    # 2. 同花顺 90 行业板块排名 — real sector-level flow data
    # ------------------------------------------------------------------
    sector_flow_records: list[dict] = []

    try:
        import akshare as ak

        df = ak.stock_board_industry_summary_ths()
        if df is not None and not df.empty:
            logger.info("[sector_flow] got %d industry sectors from akshare", len(df))

            for i, row in df.iterrows():
                name = str(row.get("板块", ""))
                if not name:
                    continue

                change_pct = float(row.get("涨跌幅", 0) or 0)
                turnover_yi = float(row.get("总成交额", 0) or 0)
                net_inflow_yi = float(row.get("净流入", 0) or 0) if "净流入" in df.columns else 0.0
                up_count = int(row.get("上涨家数", 0) or 0)
                down_count = int(row.get("下跌家数", 0) or 0)
                leader_name = str(row.get("领涨股", "") or "")
                leader_change = float(row.get("领涨股-涨跌幅", 0) or 0)

                # Convert 亿 → 万元 for frontend consistency
                main_force_net_wan = round(net_inflow_yi * 10000, 2)
                turnover_wan = round(turnover_yi * 10000, 2)

                # main_force_ratio = net_inflow / turnover
                main_force_ratio = (
                    round(net_inflow_yi / turnover_yi * 100, 2)
                    if turnover_yi > 0 else 0.0
                )

                sector_flow_records.append({
                    "sector_code": "",
                    "sector_name": name,
                    "change_pct": round(change_pct, 2),
                    "heat": up_count + down_count,
                    "stock_count": up_count + down_count,
                    "main_force_net": main_force_net_wan,
                    "main_force_ratio": main_force_ratio,
                    "super_large_net": 0.0,
                    "large_net": 0.0,
                    "leading_stock": "",
                    "leading_stock_name": leader_name,
                    "leading_stock_change": round(leader_change, 2),
                    "turnover_wan": turnover_wan,
                    "up_count": up_count,
                    "down_count": down_count,
                })

            # Sort by net inflow descending
            sector_flow_records.sort(key=lambda r: r["main_force_net"], reverse=True)

            top5 = [(r["sector_name"], r["main_force_net"]) for r in sector_flow_records[:5]]
            logger.info("[sector_flow] top 5 industries: %s", top5)
        else:
            logger.warning("[sector_flow] stock_board_industry_summary_ths returned empty")
    except Exception as exc:
        logger.warning("[sector_flow] industry summary failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Save sector flow to history cache (for trend analysis in scoring)
    # ------------------------------------------------------------------
    if sector_flow_records:
        _save_sector_flow_history(trade_date, sector_flow_records)

    return {
        "sector_tags": sector_tags,
        "watchlist": discovered_symbols,
        "ths_hot_items": ths_hot_items,
        "sector_flow_records": sector_flow_records,
    }


# ---------------------------------------------------------------------------
# Sector flow history cache — JSON file for cross-day trend analysis
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_SECTOR_HISTORY_PATH = _Path(__file__).resolve().parent.parent.parent / "data_lake" / "sector_flow_history.json"
_SECTOR_HISTORY_MAX_DAYS = 30


def _save_sector_flow_history(trade_date: str, records: list[dict]):
    """Append today's sector flow to the history JSON file. Keep last N days."""
    try:
        history: dict[str, list[dict]] = {}
        if _SECTOR_HISTORY_PATH.exists():
            history = _json.loads(_SECTOR_HISTORY_PATH.read_text(encoding="utf-8"))
        history[trade_date] = records
        # Prune old entries
        if len(history) > _SECTOR_HISTORY_MAX_DAYS:
            sorted_dates = sorted(history.keys(), reverse=True)
            for old_date in sorted_dates[_SECTOR_HISTORY_MAX_DAYS:]:
                del history[old_date]
        _SECTOR_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SECTOR_HISTORY_PATH.write_text(_json.dumps(history, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # best-effort


def load_sector_flow_history() -> dict[str, list[dict]]:
    """Load sector flow history. Returns {date: [records]}."""
    try:
        if _SECTOR_HISTORY_PATH.exists():
            return _json.loads(_SECTOR_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# Module-level singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
