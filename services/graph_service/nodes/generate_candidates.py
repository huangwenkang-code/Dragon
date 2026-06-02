"""generate_candidates — scores potential leader stocks from core 3 layers:
capital flow + THS hot concepts + dragon-tiger board.

LLM event extraction (find_news_double_layer) and FinBERT sentiment
(analyze_sentiment) are bypassed but code retained for future experiments.
"""

from shared.schemas.agent_state import AgentState, LeaderCandidate
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Core 3-layer scoring weights
W_FLOW = 0.40       # capital flow / northbound / main force
W_CONCEPT = 0.35    # THS hot concept matching
W_LHB = 0.25        # dragon-tiger board seats


async def generate_candidates(state: AgentState) -> dict:
    """Generate leader-candidate stocks from core 3 signal layers.

    Scoring weights:
      - capital flow:  0.40  (main force net, dde, northbound)
      - THS concepts:  0.35  (hot concept leader/participation)
      - dragon tiger:  0.25  (famous trader seats, net buy/sell)

    Sector flow history is used to calibrate concept_score:
      - Sectors with positive multi-week trend → concept boost
      - Sectors with negative trend → concept dampen
    """
    capital_flow_records = state.get("capital_flow_records", [])
    dragon_tiger_records = state.get("dragon_tiger_records", [])
    active_stocks = state.get("active_stocks", [])

    # Build indices by symbol
    flow_by_sym = {r.get("symbol", ""): r for r in capital_flow_records}
    lhb_by_sym = {r.get("stock_code", ""): r for r in dragon_tiger_records}

    # Build concept index from merge node's active_stocks
    concept_by_sym: dict[str, dict] = {}
    for a in active_stocks:
        sym = a.get("symbol", "")
        if sym:
            concept_by_sym[sym] = {
                "concept_score": a.get("concept_score", 0.0),
                "matched_concepts": a.get("matched_concepts", []),
            }

    # Load sector flow momentum for concept calibration
    sector_momentum = _compute_sector_momentum()
    if sector_momentum:
        logger.info("[generate_candidates] loaded sector momentum for %d sectors",
                    len(sector_momentum))

    # Collect all candidate symbols (flow + LHB + active_stocks)
    all_symbols: set[str] = set()
    all_symbols.update(flow_by_sym.keys())
    all_symbols.update(lhb_by_sym.keys())
    all_symbols.update(concept_by_sym.keys())

    if not all_symbols:
        logger.warning("[generate_candidates] no symbols to score")
        return {"leader_candidates": []}

    logger.info(
        "[generate_candidates] inputs — flow:%d lhb:%d concepts:%d → %d symbols",
        len(capital_flow_records), len(dragon_tiger_records),
        len(concept_by_sym), len(all_symbols),
    )

    # Score each symbol
    candidates: list[dict] = []
    for sym in all_symbols:
        scores = _compute_scores(sym, flow_by_sym, lhb_by_sym, concept_by_sym, sector_momentum)
        candidates.append(scores)

    # Log top 5 for debugging
    top5 = sorted(candidates, key=lambda c: c["leader_score"], reverse=True)[:5]
    for c in top5:
        logger.info("[generate_candidates] %s score=%.3f flow=%.3f concept=%.3f lhb=%.3f | %s",
                    c["stock_code"], c["leader_score"],
                    c["flow_score"], c["concept_score"], c["lhb_score"],
                    c.get("reasoning", ""))

    # Sort and rank
    candidates.sort(key=lambda c: c["leader_score"], reverse=True)
    top_n = state.get("top_n", 60)
    candidates = candidates[:top_n]
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    logger.info("[generate_candidates] produced %d candidates (top score=%.3f)",
                len(candidates), candidates[0]["leader_score"] if candidates else 0)

    return {"leader_candidates": candidates}


# ---------------------------------------------------------------------------
# Monster matching (data lake case-based reasoning)
# ---------------------------------------------------------------------------

async def _enrich_with_monster_reference(
    top_candidates: list[dict],
    flow_by_sym: dict,
    active_stocks: list[dict],
) -> None:
    """Find similar historical monsters for top candidates and attach references."""
    if not top_candidates:
        return

    try:
        from db.connection import async_session_factory
        from services.monster_matcher.matcher import MonsterMatcher, CandidateFeatures

        async with async_session_factory() as session:
            matcher = MonsterMatcher(session)

            for candidate in top_candidates:
                sym = candidate.get("stock_code", "")
                flow = flow_by_sym.get(sym, {})

                # Build candidate features from pipeline data
                features = CandidateFeatures(
                    symbol=sym,
                    stock_name=candidate.get("stock_name", ""),
                    market_cap=float(flow.get("market_cap", 0)),
                    turnover_pct=float(flow.get("turnover_pct", 0)),
                    price=float(flow.get("price", 0)),
                    change_pct=float(flow.get("change_pct", 0)),
                    sector=_find_sector_for_symbol(sym, active_stocks),
                )

                matches = await matcher.find_similar(features, top_k=3, use_l3=True)

                if matches:
                    from services.monster_matcher.matcher import _type_cn

                    ref = {
                        "top_matches": [
                            {
                                "stock_code": m.monster.stock_code,
                                "stock_name": m.monster.stock_name,
                                "similarity": round(m.similarity_score, 3),
                                "primary_type": _type_cn(m.monster.primary_type),
                                "max_gain_pct": float(m.monster.max_gain_pct),
                                "trading_days": m.monster.trading_days,
                                "match_reasons": m.match_reasons,
                            }
                            for m in matches
                        ],
                        "summary": _generate_monster_summary(features, matches),
                    }
                    candidate["monster_reference"] = ref
                    logger.info(
                        "[generate_candidates] monster_match %s → top=%s(%.1f%%)",
                        sym,
                        matches[0].monster.stock_name,
                        matches[0].similarity_score * 100,
                    )
    except Exception as exc:
        logger.warning("[generate_candidates] monster matching skipped: %s", exc)


def _find_sector_for_symbol(sym: str, active_stocks: list[dict]) -> str:
    """Find sector for a symbol from active_stocks data."""
    for a in active_stocks:
        if a.get("symbol") == sym:
            concepts = a.get("matched_concepts", [])
            if concepts:
                return ", ".join(concepts[:3])
            return a.get("reasons", "")
    return ""


def _generate_monster_summary(
    features: "CandidateFeatures",
    matches: list["MatchResult"],
) -> str:
    """Generate a brief analogy summary for the top match."""
    if not matches:
        return "未找到相似历史妖股。"
    from services.monster_matcher.matcher import _type_cn

    top = matches[0]
    m = top.monster
    reasons = "; ".join(top.match_reasons) if top.match_reasons else "多维特征相似"

    return (
        f"历史相似案例: {m.stock_name}({m.stock_code}) "
        f"相似度{top.similarity_score:.0%}，"
        f"于{m.trading_days}天内涨幅+{m.max_gain_pct}%，"
        f"类型: {_type_cn(m.primary_type)}/{_type_cn(m.secondary_type)}。"
        f"匹配理由: {reasons}。"
    )


# ---------------------------------------------------------------------------
# Sector momentum from history cache
# ---------------------------------------------------------------------------

def _compute_sector_momentum() -> dict[str, float]:
    """Compute per-sector momentum from sector flow history.

    Loads sector_flow_history.json, calculates cumulative net inflow
    over 1-week and 4-week windows, returns a momentum score per sector.

    Momentum score range roughly [-2, +2]:
      +2 = strong sustained inflow (both 1w and 4w positive)
       0 = flat / no data
      -2 = strong sustained outflow
    """
    try:
        from services.graph_service.graph import load_sector_flow_history
        history = load_sector_flow_history()
    except Exception:
        return {}

    if not history:
        return {}

    # Aggregate net inflow by sector across all dates
    from collections import defaultdict
    from datetime import date as dt_date, timedelta

    sector_daily: dict[str, dict[str, float]] = defaultdict(dict)
    all_dates = sorted(history.keys())

    for d in all_dates:
        records = history.get(d, [])
        if not isinstance(records, list):
            continue
        for rec in records:
            name = rec.get("sector_name", "")
            if not name:
                continue
            net = rec.get("main_force_net", 0) or 0
            sector_daily[name][d] = net

    if not all_dates:
        return {}

    today = dt_date.today()
    one_week_ago = (today - timedelta(days=7)).isoformat()
    four_weeks_ago = (today - timedelta(days=28)).isoformat()

    momentum: dict[str, float] = {}
    for sector_name, daily in sector_daily.items():
        dates = sorted(daily.keys())
        # 1-week trend
        recent_1w = [daily[d] for d in dates if d >= one_week_ago]
        # 4-week trend
        recent_4w = [daily[d] for d in dates if d >= four_weeks_ago]

        w1 = sum(recent_1w) / max(len(recent_1w), 1)
        w4 = sum(recent_4w) / max(len(recent_4w), 1)

        # Normalize by max observed net inflow for scoring
        # Score: direction × magnitude
        def _norm(val, abs_max):
            if abs_max < 1:
                return 0.0
            return max(-2.0, min(2.0, val / (abs_max / 2)))

        # Combined momentum: 1w has more weight (60%) than 4w baseline (40%)
        all_vals = [v for v in daily.values()]
        abs_max = max(abs(v) for v in all_vals) if all_vals else 1.0
        w1_norm = _norm(w1, abs_max)
        w4_norm = _norm(w4, abs_max)
        momentum[sector_name] = round(w1_norm * 0.6 + w4_norm * 0.4, 2)

    return momentum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_sentiment(scores: list[dict]) -> dict[str, dict]:
    """Aggregate sentiment scores by symbol."""
    by_sym: dict[str, dict] = {}
    for s in scores:
        sym = s.get("symbol", s.get("target_id", ""))
        if not sym:
            continue
        if sym not in by_sym:
            by_sym[sym] = {
                "sentiment": [], "narrative": [], "hype": [],
                "consensus": [], "risk": [], "diffusion": [],
                "keywords": [],
            }
        d = by_sym[sym]
        d["sentiment"].append(s.get("sentiment_score", 0))
        d["narrative"].append(s.get("narrative_score", s.get("narrative_strength", 0)))
        d["hype"].append(s.get("hype_score", 0))
        d["consensus"].append(s.get("consistency_score", s.get("consensus", 0.5)))
        d["risk"].append(s.get("risk_score", 0))
        d["diffusion"].append(s.get("diffusion_speed", 0))
        d["keywords"].extend(s.get("keywords", []))
    return by_sym


def _compute_scores(
    sym: str,
    flow_by_sym: dict,
    lhb_by_sym: dict,
    concept_by_sym: dict,
    sector_momentum: dict[str, float] | None = None,
) -> dict:
    """Compute leader_score from core 3 layers: flow + concept + LHB.

    Weights (empirical, calibratable):
      - capital flow:  0.40  (main force net, dde, northbound)
      - THS concepts:  0.35  (hot concept leader/participation)
      - dragon tiger:  0.25  (famous trader seats, net buy/sell)

    Sector momentum calibration: concept_score × momentum_multiplier.
    """

    # -- Capital flow sub-score --
    flow = flow_by_sym.get(sym, {})
    flow_score = flow.get("flow_score", 0.0)
    flow_net = flow.get("main_force_net", 0.0)
    dde_val = flow.get("main_force_ratio", 0.0)

    # -- Dragon tiger sub-score --
    lhb = lhb_by_sym.get(sym, {})
    lhb_score = lhb.get("lhb_score", 0.0)
    trader_signal = lhb.get("trader_signal", "")
    has_trader = bool(lhb.get("famous_traders", []))

    # -- THS concept sub-score (with sector momentum calibration) --
    concept_data = concept_by_sym.get(sym, {})
    concept_score = concept_data.get("concept_score", 0.0)
    matched_concepts = concept_data.get("matched_concepts", [])

    # Apply sector momentum multiplier to concept_score
    momentum_mult = 1.0
    if sector_momentum and matched_concepts:
        momentum_vals = []
        for mc in matched_concepts:
            m = sector_momentum.get(mc, 0.0) or sector_momentum.get(mc.replace("概念", ""), 0.0)
            if m != 0.0:
                momentum_vals.append(m)
        if momentum_vals:
            avg_momentum = sum(momentum_vals) / len(momentum_vals)
            # Map momentum [-2, +2] → multiplier [0.70, 1.30]
            momentum_mult = round(1.0 + avg_momentum * 0.15, 2)
            momentum_mult = max(0.70, min(1.30, momentum_mult))
    concept_score = round(concept_score * momentum_mult, 3)
    concept_score = min(1.0, concept_score)

    # -- Composite leader_score (3 layers) --
    leader_score = (
        flow_score * W_FLOW
        + concept_score * W_CONCEPT
        + lhb_score * W_LHB
    )
    leader_score = max(0.0, min(1.0, leader_score))

    # -- Monster potential (simplified: LHB-driven) --
    monster = (
        lhb_score * 0.50
        + (0.30 if has_trader else 0)
        + (0.20 if trader_signal == "合力做多" else 0)
    )
    monster = max(0.0, min(1.0, monster))

    # -- Reasoning --
    reasons = []
    if dde_val > 0:
        reasons.append(f"大单净量{dde_val:.2f}%")
    if flow_net > 0:
        reasons.append(f"主力净流入{flow_net:.0f}万")
    elif flow_net < -100:
        reasons.append(f"主力净流出{abs(flow_net):.0f}万")
    if flow_score > 0.5:
        reasons.append(f"成交活跃 {flow.get('amount_wan', 0):.0f}万")
    if matched_concepts:
        reasons.append(f"热点概念({','.join(matched_concepts[:3])})")
    if trader_signal == "合力做多":
        reasons.append(f"游资合力做多({','.join(lhb.get('famous_traders', [])[:2])})")
    elif trader_signal == "分歧":
        reasons.append("游资分歧")
    elif trader_signal == "出货":
        reasons.append("游资出货")
    elif has_trader:
        reasons.append(f"知名游资({','.join(lhb.get('famous_traders', [])[:2])})")
    if not reasons:
        reasons.append("综合数据不足")

    return {
        "stock_code": sym,
        "stock_name": flow.get("stock_name", lhb.get("stock_name", sym)),
        "leader_score": round(leader_score, 4),
        "dragon_score": round(leader_score, 4),
        "flow_score": round(flow_score, 3),
        "concept_score": round(concept_score, 3),
        "lhb_score": round(lhb_score, 3),
        "monster_potential": round(monster, 3),
        "limit_up_prob": 0.0,
        "reasoning": "; ".join(reasons),
        "sector": "",
        "rank": 0,
        "event_sub": 0.0,
        "flow_sub": round(flow_score * W_FLOW, 4),
        "concept_sub": round(concept_score * W_CONCEPT, 4),
        "lhb_sub": round(lhb_score * W_LHB, 4),
    }


def _try_model_ensemble(state: AgentState) -> dict[str, float]:
    """Try to run ML model ensemble predictions.

    Returns a dict of symbol -> ensemble_score (0-1), or empty dict
    if models are not trained / not available.
    """
    watchlist = state.get("watchlist", [])
    if not watchlist:
        return {}

    try:
        from services.model_engine.ensemble import ModelEnsemble
        from services.model_engine.base import BaseModel

        ensemble = ModelEnsemble()

        # Try to load each model type
        _try_load_model(ensemble, "xgboost", "services.model_engine.xgboost_model", "XGBModel")
        _try_load_model(ensemble, "gats", "services.model_engine.gats_model", "GATModel")

        if not ensemble.is_fitted:
            logger.debug("[generate_candidates] no ML models fitted — skipping ensemble")
            return {}

        # Build minimal feature frame from state data
        import pandas as pd
        features = _build_feature_frame(state)
        if features.empty:
            return {}

        scores = ensemble.predict_single(features)
        return scores.to_dict()

    except Exception as exc:
        logger.warning("[generate_candidates] model ensemble skipped: %s", exc)
        return {}


def _try_load_model(ensemble, name: str, module_path: str, class_name: str) -> None:
    """Try to import and register a model, silently skip if unavailable."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return
        instance = cls()
        if hasattr(instance, 'model') and instance.model is not None:
            ensemble.add_model(name, instance)
    except Exception:
        pass


def _build_feature_frame(state: AgentState):
    """Build a minimal feature DataFrame from pipeline state.

    Uses available numeric signals as features for ML models.
    """
    import pandas as pd

    flow_records = state.get("capital_flow_records", [])
    lhb_records = state.get("dragon_tiger_records", [])
    sentiment_scores = state.get("sentiment_scores", [])

    rows = []
    symbols = set()
    for r in flow_records:
        symbols.add(r.get("symbol", ""))
    for r in lhb_records:
        symbols.add(r.get("stock_code", ""))
    for s in sentiment_scores:
        sym = s.get("symbol", s.get("target_id", ""))
        if sym:
            symbols.add(sym)

    sent_by_sym = _index_sentiment(sentiment_scores)
    flow_by_sym = {r.get("symbol", ""): r for r in flow_records}
    lhb_by_sym = {r.get("stock_code", ""): r for r in lhb_records}

    for sym in symbols:
        if not sym:
            continue
        s_data = sent_by_sym.get(sym, {})
        f_data = flow_by_sym.get(sym, {})
        l_data = lhb_by_sym.get(sym, {})

        s_list = s_data.get("sentiment", [])
        h_list = s_data.get("hype", [])
        r_list = s_data.get("risk", [])

        rows.append({
            "symbol": sym,
            "sentiment_mean": sum(s_list) / len(s_list) if s_list else 0.0,
            "hype_mean": sum(h_list) / len(h_list) if h_list else 0.0,
            "risk_mean": sum(r_list) / len(r_list) if r_list else 0.0,
            "flow_score": f_data.get("flow_score", 0.0),
            "flow_net": f_data.get("main_force_net", 0.0),
            "flow_ratio": f_data.get("flow_ratio", 0.0),
            "lhb_score": l_data.get("lhb_score", 0.0),
            "lhb_net": l_data.get("net_amount", 0.0),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.set_index("symbol", inplace=True)
    return df.fillna(0.0)
