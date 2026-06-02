"""
Minimal multi-factor scorer — 5 factors with clear economic rationale.

Design principles:
  1. Every factor has a causal story that holds across market cycles
  2. Percentile-rank normalization → uniform 0-1, no max-division clustering
  3. Equal weights by default → no regime overfitting
  4. Transparent: every score is explainable

Factors:
  F1. liquidity    — higher daily turnover = less crash risk (timeless)
  F2. concept      — is this stock a 同花顺 concept leader? (theme persistence)
  F3. oversold     — further from 60d high = more bounce potential (mean reversion)
  F4. not_extended — lower abs(chg%) today = less reversal risk (momentum reversal)
  F5. not_exhausted — moderate turnover% = not a volume climax (exhaustion)

All factors are percentile-ranked within the daily candidate pool.
Final score = simple average of the 5 percentile ranks.
"""

from __future__ import annotations

import math

from shared.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_float(val, default=0.0) -> float:
    if val is None: return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _percentile_rank(values: list[float]) -> list[float]:
    """Convert raw values to [0, 1] percentile ranks. Higher raw = higher rank."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for pos, idx in enumerate(sorted_idx):
        ranks[idx] = pos / (n - 1)
    return ranks


def _inverse_rank(values: list[float]) -> list[float]:
    """Percentile rank where LOWER raw = HIGHER rank."""
    return _percentile_rank([-v for v in values])


def _peak_rank(values: list[float]) -> list[float]:
    """Percentile rank where MIDDLE values score highest, extremes score lowest.
    f(x) = 1 - 2*|x - 0.5|  after percentile ranking.
    """
    pr = _percentile_rank(values)
    return [1.0 - 2.0 * abs(r - 0.5) for r in pr]


def score_candidates(
    candidates: list[dict],
    trade_date: str = "",
    regime: str = "CHOPPY",  # kept for API compatibility, currently unused
) -> list[dict]:
    """Score all candidates with 5-factor model. Returns sorted list.

    Each candidate must have:
      - symbol / stock_code
      - amount (daily turnover in yuan)
      - change_pct
      - turnover_pct
    Optional:
      - _source ('concept_leader' → activates F2)
      - drawdown_pct (from oversold query)
      - price
    """
    n = len(candidates)
    if n == 0:
        return candidates

    # ---- Extract raw factor values ----
    f1_raw = [_safe_float(c.get("amount", 0)) for c in candidates]                                    # liquidity
    f2_raw = [1.0 if c.get("_source") == "concept_leader" else 0.0 for c in candidates]              # concept
    f3_raw = [_safe_float(c.get("drawdown_pct", 0)) for c in candidates]                               # oversold
    f4_raw = [abs(_safe_float(c.get("change_pct", 0))) for c in candidates]                            # extended (raw, to be inverted)
    f5_raw = [_safe_float(c.get("turnover_pct", 0)) for c in candidates]                               # turnover (raw, peak-ranked)

    # ---- Percentile-rank each factor ----
    f1_rank = _percentile_rank(f1_raw)       # F1: more liquid = better
    f2_rank = _percentile_rank(f2_raw)       # F2: concept leader = better
    f3_rank = _percentile_rank(f3_raw)       # F3: more oversold = better
    f4_rank = _inverse_rank(f4_raw)          # F4: LESS extended = better (invert!)
    f5_rank = _peak_rank(f5_raw)             # F5: moderate turnover = best, extremes = worst

    # ---- Compute final score: equal-weight average ----
    for i, c in enumerate(candidates):
        score = (f1_rank[i] + f2_rank[i] + f3_rank[i] + f4_rank[i] + f5_rank[i]) / 5.0
        c["leader_score"] = round(score, 4)

        # Build explainable reasoning
        parts = []
        if f1_rank[i] > 0.6: parts.append("流动性强")
        elif f1_rank[i] < 0.3: parts.append("流动性弱")
        if f2_rank[i] > 0.5: parts.append("概念龙头")
        if f3_rank[i] > 0.6: parts.append(f"超跌{_safe_float(c.get('drawdown_pct',0)):.0f}%")
        if f4_rank[i] > 0.6: parts.append("未透支")
        elif f4_rank[i] < 0.3: parts.append(f"涨{abs(_safe_float(c.get('change_pct',0))):.1f}%追高风险")
        if f5_rank[i] < 0.3: parts.append("换手极端")
        c["reasoning"] = "; ".join(parts) if parts else "综合均衡"

    candidates.sort(key=lambda c: c.get("leader_score", 0), reverse=True)
    return candidates
