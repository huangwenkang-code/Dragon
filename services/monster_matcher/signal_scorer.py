"""Rules-based monster signal scorer.

Transparent 6-dimension scoring that replaces the zero-valued ml_sub.
No training — uses known monster stock patterns: small cap, trader
participation, turnover explosion, sentiment/hype anomalies, concept
heat, limit-up streaks.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Speculative keywords for limit-up streak estimation
SPEC_KW = {"连板", "龙头", "妖股", "打板", "接力", "涨停", "热点"}

# 6-dimension weights, sum to 1.0
DEFAULT_WEIGHTS = {
    "small_cap": 0.192,
    "trader_participation": 0.192,
    "turnover_explosion": 0.20,
    "sentiment_anomaly": 0.152,
    "concept_heat": 0.152,
    "limit_up_streak": 0.112,
}


@dataclass
class SignalBreakdown:
    """Per-dimension scores for auditability."""
    small_cap: float = 0.0
    trader_participation: float = 0.0
    turnover_explosion: float = 0.0
    sentiment_anomaly: float = 0.0
    concept_heat: float = 0.0
    limit_up_streak: float = 0.0

    def as_dict(self) -> dict:
        return {
            "small_cap": round(self.small_cap, 3),
            "trader_participation": round(self.trader_participation, 3),
            "turnover_explosion": round(self.turnover_explosion, 3),
            "sentiment_anomaly": round(self.sentiment_anomaly, 3),
            "concept_heat": round(self.concept_heat, 3),
            "limit_up_streak": round(self.limit_up_streak, 3),
        }


def score_stock(
    market_cap: float = 0.0,
    turnover_pct: float = 0.0,
    famous_traders: list[str] | None = None,
    lhb_score: float = 0.0,
    avg_sentiment: float = 0.0,
    avg_hype: float = 0.0,
    matched_concepts: list[str] | None = None,
    sentiment_keywords: list[str] | None = None,
    event_keywords: list[str] | None = None,
) -> tuple[float, SignalBreakdown]:
    """Compute the monster signal score (0-1) for a single stock.

    Args:
        market_cap: Market cap in 亿.
        turnover_pct: Recent daily turnover % (换手率).
        famous_traders: List of famous trader names from dragon tiger board.
        lhb_score: Raw dragon tiger score.
        avg_sentiment: Average sentiment score from FinBERT.
        avg_hype: Average hype score.
        matched_concepts: List of concept names matched for this stock.
        sentiment_keywords: Keywords from sentiment analysis.
        event_keywords: Keywords from related events.

    Returns:
        (total_score, SignalBreakdown) — both 0-1.
    """
    breakdown = SignalBreakdown()
    available: set[str] = set()

    # --- Small Cap ---
    if market_cap > 0:
        if market_cap < 50:
            breakdown.small_cap = 1.0
        elif market_cap < 100:
            breakdown.small_cap = 0.7
        elif market_cap < 200:
            breakdown.small_cap = 0.3
        else:
            breakdown.small_cap = 0.0
        available.add("small_cap")

    # --- Turnover Explosion (换手爆发) ---
    if turnover_pct > 0:
        if turnover_pct > 15:
            breakdown.turnover_explosion = 1.0
        elif turnover_pct > 10:
            breakdown.turnover_explosion = 0.7
        elif turnover_pct > 5:
            breakdown.turnover_explosion = 0.4
        elif turnover_pct > 2:
            breakdown.turnover_explosion = 0.2
        else:
            breakdown.turnover_explosion = 0.0
        available.add("turnover_explosion")

    # --- Trader Participation ---
    traders = famous_traders or []
    if traders:
        breakdown.trader_participation = 1.0
        available.add("trader_participation")
    elif lhb_score > 0:
        breakdown.trader_participation = 0.5
        available.add("trader_participation")
    else:
        breakdown.trader_participation = 0.0

    # --- Sentiment Anomaly ---
    has_sentiment = avg_sentiment > 0 or avg_hype > 0
    if avg_sentiment > 0.8 and avg_hype > 0.3:
        breakdown.sentiment_anomaly = 1.0
    elif avg_sentiment > 0.5:
        breakdown.sentiment_anomaly = 0.5
    elif has_sentiment:
        breakdown.sentiment_anomaly = 0.2
    else:
        breakdown.sentiment_anomaly = 0.0
    if has_sentiment:
        available.add("sentiment_anomaly")

    # --- Concept Heat ---
    has_concepts = matched_concepts is not None and len(matched_concepts) > 0
    concepts = matched_concepts or []
    n = len(concepts)
    if n >= 3:
        breakdown.concept_heat = 1.0
    elif n == 2:
        breakdown.concept_heat = 0.7
    elif n == 1:
        breakdown.concept_heat = 0.4
    else:
        breakdown.concept_heat = 0.0
    if has_concepts:
        available.add("concept_heat")

    # --- Limit-Up Streak ---
    all_keywords = set(sentiment_keywords or []) | set(event_keywords or [])
    kw_matches = len(all_keywords & SPEC_KW)
    breakdown.limit_up_streak = min(kw_matches / 3.0, 1.0)
    if kw_matches > 0:
        available.add("limit_up_streak")

    # Redistribute weights: drop unavailable dimensions, re-normalize
    active_weights = {
        dim: DEFAULT_WEIGHTS[dim]
        for dim in available
        if dim in DEFAULT_WEIGHTS
    }
    total_weight = sum(active_weights.values()) if active_weights else 1.0

    if total_weight == 0:
        return 0.0, breakdown

    score = sum(
        getattr(breakdown, dim) * (active_weights.get(dim, 0) / total_weight)
        for dim in active_weights
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return score, breakdown
