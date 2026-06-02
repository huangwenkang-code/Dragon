"""merge_active_stocks — merge capital flow + concept heat + dragon-tiger signals.

Produces a ranked candidate pool of stocks with unusual activity.
Weights: capital_flow 0.40, concept_heat 0.35, dragon_tiger 0.15, residual 0.10

Enrichment sources:
  - 同花顺热点 ddejingliang (大单净量%, 72 hot stocks)
  - 同花顺 90 industry sector flow (real sector-level net inflow, 亿)
"""

import asyncio as _asyncio
import time as _time

from shared.schemas.agent_state import AgentState
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Merge weights
W_FLOW = 0.40
W_CONCEPT = 0.35
W_LHB = 0.15


async def merge_active_stocks(state: AgentState) -> dict:
    """Merge upstream signals into a ranked active-stock pool.

    Inputs:
      - capital_flow_records: ranked by activity (volume_ratio × change)
      - sector_flow_records: real industry sector data from sector_flow node
      - sector_tags: hot concepts from 同花顺热点
      - dragon_tiger_records: stocks on today's LHB

    Outputs:
      - active_stocks: merged & scored candidate pool (top 30)
      - capital_flow_records: enriched with ddejingliang-based main_force_net
      - capital_flow_summary: aggregate stats for frontend
      - sector_flow_records: pass-through (real industry data)
    """
    flow_records = state.get("capital_flow_records", [])
    sector_flow = state.get("sector_flow_records", [])
    sector_tags = state.get("sector_tags", [])
    lhb_records = state.get("dragon_tiger_records", [])
    ths_hot_items = state.get("ths_hot_items", [])

    logger.info(
        "[merge] inputs — flow:%d sector_flow:%d concepts:%d lhb:%d ths_hot:%d",
        len(flow_records), len(sector_flow), len(sector_tags), len(lhb_records),
        len(ths_hot_items),
    )

    # ------------------------------------------------------------------
    # 1. 同花顺热点 dde 富化 — ddejingliang → main_force_ratio
    # ------------------------------------------------------------------
    dde_by_sym: dict[str, dict] = {}
    for item in ths_hot_items:
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        dde_by_sym[code] = {
            "ddejingliang": float(item.get("ddejingliang", 0) or 0),
            "chengjiaoe": float(item.get("chengjiaoe", 0) or 0),
            "zhangfu": float(item.get("zhangfu", 0) or 0),
            "huanshou": float(item.get("huanshou", 0) or 0),
        }

    enriched_count = 0
    for r in flow_records:
        sym = r.get("symbol", "")
        dde = dde_by_sym.get(sym)
        if dde:
            r["main_force_ratio"] = round(dde["ddejingliang"], 2)
            r["amount_wan"] = round(dde["chengjiaoe"], 2)
            r["_source"] = "ths_hot"
            enriched_count += 1
    if enriched_count:
        logger.info("[merge] enriched %d/%d flow records with 同花顺 ddejingliang",
                    enriched_count, len(flow_records))

    # Add 同花顺 hot stocks not already in flow pool
    existing_syms = {r["symbol"] for r in flow_records}
    added_from_ths = 0
    for item in ths_hot_items:
        code = str(item.get("code", "")).strip()
        if not code or code in existing_syms:
            continue
        name = str(item.get("name", "")).strip()
        dde = dde_by_sym.get(code, {})
        flow_records.append({
            "symbol": code,
            "stock_name": name,
            "price": float(item.get("close", 0) or 0),
            "change_pct": float(item.get("zhangfu", 0) or 0),
            "amount": 0,
            "amount_wan": round(dde.get("chengjiaoe", 0), 2),
            "main_force_net": 0.0,
            "main_force_ratio": round(dde.get("ddejingliang", 0), 2),
            "super_large_net": 0.0,
            "large_net": 0.0,
            "mid_net": 0.0,
            "small_net": 0.0,
            "total_net": 0.0,
            "northbound_net": 0.0,
            "flow_ratio": 0.0,
            "flow_score": 0.0,
            "sector_flow": 0.0,
            "pe": 0,
            "pb": 0,
            "market_cap": 0,
            "_source": "ths_hot",
        })
        added_from_ths += 1
    if added_from_ths:
        logger.info("[merge] added %d stocks from 同花顺热点 to flow pool", added_from_ths)

    # ------------------------------------------------------------------
    # 2. Compute main_force_net from 同花顺 ddejingliang (no EastMoney needed)
    #    main_force_net(万元) = amount_wan × ddejingliang / 100
    # ------------------------------------------------------------------
    for r in flow_records:
        dde_pct = r.get("main_force_ratio", 0) or 0
        amount_wan = r.get("amount_wan", 0) or 0
        if dde_pct != 0 and amount_wan > 0:
            net = round(amount_wan * dde_pct / 100, 2)
            r["main_force_net"] = net
            r["total_net"] = net
            r["super_large_net"] = 0.0
            r["large_net"] = 0.0
            r["mid_net"] = 0.0
            r["small_net"] = 0.0
            r["total_net"] = net

    # ------------------------------------------------------------------
    # 3. Build indices & score
    # ------------------------------------------------------------------
    flow_by_sym = {r["symbol"]: r for r in flow_records}
    lhb_by_sym = {r.get("stock_code", ""): r for r in lhb_records}

    concept_leaders: dict[str, str] = {}
    for tag in sector_tags:
        name = tag.get("concept_name", "") if isinstance(tag, dict) else str(tag)
        leader = tag.get("leader_stock", "") if isinstance(tag, dict) else ""
        if name and leader:
            concept_leaders[name] = leader

    all_symbols: set[str] = set()
    all_symbols.update(flow_by_sym.keys())
    all_symbols.update(lhb_by_sym.keys())
    all_symbols.update(concept_leaders.values())
    for s in sector_flow:
        ls = s.get("leading_stock", "")
        if ls:
            all_symbols.add(ls)

    if not all_symbols:
        logger.warning("[merge] no active stocks discovered")
        return {"active_stocks": []}

    # Normalize flow_score
    max_dde = max(
        (r.get("main_force_ratio", 0) for r in flow_records if r.get("main_force_ratio", 0) > 0),
        default=0,
    )
    for r in flow_records:
        dde = r.get("main_force_ratio", 0)
        if dde > 0 and max_dde > 0:
            r["flow_score"] = round(dde / max_dde, 3)
        else:
            r["flow_score"] = r.get("flow_score", 0)
    flow_by_sym = {r["symbol"]: r for r in flow_records}

    # Score each candidate
    active: list[dict] = []
    for sym in all_symbols:
        score = 0.0
        reasons: list[str] = []

        flow = flow_by_sym.get(sym, {})
        flow_score = flow.get("flow_score", 0)
        main_net = flow.get("main_force_net", 0)
        dde_val = flow.get("main_force_ratio", 0)
        super_net = flow.get("super_large_net", 0)
        large_net = flow.get("large_net", 0)

        if flow_score > 0 or main_net != 0:
            score += flow_score * W_FLOW
            if dde_val > 0 and main_net != 0:
                reasons.append(f"大单净量{dde_val:.2f}% 净流入{main_net:.0f}万")
            elif main_net > 0:
                reasons.append(f"主力净流入{main_net:.0f}万")
            elif main_net < -5000:
                reasons.append(f"主力大幅流出{abs(main_net):.0f}万")
            elif flow.get("amount_wan", 0) > 0:
                reasons.append(f"成交活跃 {flow.get('amount_wan', 0):.0f}万")

        concept_score = 0.0
        matched_concepts = []
        for c_name, c_leader in concept_leaders.items():
            if c_leader == sym:
                concept_score = max(concept_score, 0.8)
                matched_concepts.append(c_name)
            elif sym in flow_by_sym:
                concept_score = max(concept_score, 0.3)
        score += concept_score * W_CONCEPT
        if matched_concepts:
            reasons.append(f"概念龙头({','.join(matched_concepts[:3])})")

        lhb = lhb_by_sym.get(sym, {})
        lhb_score = lhb.get("lhb_score", 0)
        if lhb_score > 0:
            score += lhb_score * W_LHB
            trader_signal = lhb.get("trader_signal", "")
            if trader_signal:
                reasons.append(f"龙虎榜{trader_signal}")

        active.append({
            "symbol": sym,
            "stock_name": flow.get("stock_name", lhb.get("stock_name", sym)),
            "active_score": round(score, 4),
            "flow_score": round(flow_score, 3),
            "concept_score": round(concept_score, 3),
            "lhb_score": round(lhb_score, 3),
            "main_force_net": main_net,
            "ddejingliang": dde_val,
            "super_large_net": super_net,
            "large_net": large_net,
            "mid_net": flow.get("mid_net", 0),
            "small_net": flow.get("small_net", 0),
            "amount_wan": flow.get("amount_wan", 0),
            "change_pct": flow.get("change_pct", 0),
            "pe": flow.get("pe", 0),
            "pb": flow.get("pb", 0),
            "market_cap": flow.get("market_cap", 0),
            "_source": flow.get("_source", ""),
            "reasons": "; ".join(reasons) if reasons else "综合数据不足",
            "matched_concepts": matched_concepts,
        })

    active.sort(key=lambda x: x["active_score"], reverse=True)
    active = active[:30]
    for i, a in enumerate(active):
        a["rank"] = i + 1

    logger.info("[merge] produced %d active stocks (top: %s score=%.4f)",
                len(active), active[0]["symbol"] if active else "none",
                active[0]["active_score"] if active else 0)

    # Use real sector_flow_records from sector_flow node, fallback to concept tags
    if not sector_flow:
        sector_flow = _build_sector_flow_from_tags(sector_tags, flow_by_sym)

    # Summary
    summary = state.get("capital_flow_summary", {})
    dde_stocks = [r for r in flow_records if r.get("main_force_ratio", 0) != 0]
    summary["total_main_inflow"] = round(
        sum(r.get("main_force_net", 0) for r in dde_stocks), 2
    )
    summary["top_sectors"] = [s["sector_name"] for s in sector_flow[:5]]
    summary["dde_stocks"] = len(dde_stocks)
    summary["total_pool"] = len(flow_records)

    return {
        "active_stocks": active,
        "capital_flow_records": flow_records,
        "capital_flow_summary": summary,
        "sector_flow_records": sector_flow,
    }


# ---------------------------------------------------------------------------
# Fallback: build sector_flow from concept tags
# ---------------------------------------------------------------------------

def _build_sector_flow_from_tags(
    sector_tags: list[dict], flow_by_sym: dict
) -> list[dict]:
    """Fallback: aggregate concept tags by name using leader stock flow data."""
    if not sector_tags:
        return []

    by_name: dict[str, dict] = {}
    for tag in sector_tags:
        if not isinstance(tag, dict):
            continue
        name = str(tag.get("concept_name", ""))
        if not name:
            continue
        leader = str(tag.get("leader_stock", ""))
        flow = flow_by_sym.get(leader, {})

        if name not in by_name:
            by_name[name] = {
                "sector_code": str(tag.get("concept_id", "")),
                "sector_name": name,
                "change_pct": 0.0,
                "heat": 0,
                "stock_count": 0,
                "main_force_net": 0.0,
                "main_force_ratio": 0.0,
                "super_large_net": 0.0,
                "large_net": 0.0,
                "leading_stock": leader,
                "leading_stock_name": str(tag.get("leader_stock_name", "")),
                "leading_stock_change": float(tag.get("leader_stock_change", 0)),
            }
        by_name[name]["heat"] += 1
        by_name[name]["stock_count"] += 1
        by_name[name]["main_force_net"] += flow.get("main_force_net", 0)
        by_name[name]["main_force_ratio"] = max(
            by_name[name]["main_force_ratio"],
            flow.get("main_force_ratio", 0),
        )
        by_name[name]["super_large_net"] += flow.get("super_large_net", 0)
        by_name[name]["large_net"] += flow.get("large_net", 0)

    for entry in by_name.values():
        entry["main_force_net"] = round(entry["main_force_net"], 2)
        entry["super_large_net"] = round(entry["super_large_net"], 2)
        entry["large_net"] = round(entry["large_net"], 2)

    records = list(by_name.values())
    records.sort(key=lambda r: r["heat"], reverse=True)
    return records
