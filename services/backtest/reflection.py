"""Reflection Layer — post-backtest pattern analysis.

Identifies losing patterns and surfaces actionable insights.
Phase 2 v1: descriptive statistics + pattern flagging.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from services.backtest.models import Trade


@dataclass
class ReflectionReport:
    """Structured post-backtest analysis."""
    total_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float

    # Exit reason breakdown
    exit_reason_stats: dict = field(default_factory=dict)

    # Holding period breakdown
    holding_stats: dict = field(default_factory=dict)

    # Score bracket breakdown
    score_stats: dict = field(default_factory=dict)

    # Flagged patterns (win rate < 40%, >= 3 trades)
    flags: list[dict] = field(default_factory=list)


def analyze(trades: list[Trade]) -> ReflectionReport:
    """Run reflection analysis on completed trades."""
    if not trades:
        return ReflectionReport(0, 0, 0, 0)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    wr = len(wins) / len(trades)
    avg_pnl = sum(t.net_pnl for t in trades) / len(trades)
    total_pnl = sum(t.net_pnl for t in trades)

    # ── Exit reason breakdown ──
    exit_stats: dict[str, dict] = {}
    exit_groups = defaultdict(list)
    for t in trades:
        reason_type = t.exit_reason.split("(")[0].strip() if t.exit_reason else "unknown"
        exit_groups[reason_type].append(t)
    for reason, group in exit_groups.items():
        w = sum(1 for t in group if t.net_pnl > 0)
        exit_stats[reason] = {
            "count": len(group),
            "win_rate": round(w / len(group), 3),
            "avg_pnl": round(sum(t.net_pnl for t in group) / len(group), 0),
            "total_pnl": round(sum(t.net_pnl for t in group), 0),
        }

    # ── Holding period breakdown ──
    holding_stats: dict[str, dict] = {}
    holding_groups = defaultdict(list)
    for t in trades:
        d = t.holding_days
        if d <= 1:
            bucket = "1天"
        elif d <= 3:
            bucket = "2-3天"
        elif d <= 7:
            bucket = "4-7天"
        elif d <= 15:
            bucket = "8-15天"
        else:
            bucket = "15天+"
        holding_groups[bucket].append(t)
    for bucket, group in holding_groups.items():
        w = sum(1 for t in group if t.net_pnl > 0)
        holding_stats[bucket] = {
            "count": len(group),
            "win_rate": round(w / len(group), 3),
            "avg_pnl": round(sum(t.net_pnl for t in group) / len(group), 0),
        }

    # ── Score bracket breakdown ──
    score_stats: dict[str, dict] = {}
    score_groups = defaultdict(list)
    for t in trades:
        s = t.entry_score
        if s > 0.8:
            bucket = "高(>0.8)"
        elif s > 0.6:
            bucket = "中高(0.6-0.8)"
        elif s > 0.4:
            bucket = "中(0.4-0.6)"
        else:
            bucket = "低(<0.4)"
        score_groups[bucket].append(t)
    for bucket, group in score_groups.items():
        w = sum(1 for t in group if t.net_pnl > 0)
        score_stats[bucket] = {
            "count": len(group),
            "win_rate": round(w / len(group), 3),
            "avg_pnl": round(sum(t.net_pnl for t in group) / len(group), 0),
        }

    # ── Flag patterns with low win rate ──
    flags = []
    for reason, stats in exit_stats.items():
        if stats["count"] >= 3 and stats["win_rate"] < 0.4:
            flags.append({
                "type": "exit_reason",
                "pattern": f'退出原因"{reason}"胜率仅{stats["win_rate"]:.0%}',
                "detail": f'{stats["count"]}笔，均亏¥{stats["avg_pnl"]:,.0f}',
            })
    for bucket, stats in holding_stats.items():
        if stats["count"] >= 3 and stats["win_rate"] < 0.4:
            flags.append({
                "type": "holding_period",
                "pattern": f"持有{bucket}胜率仅{stats['win_rate']:.0%}",
                "detail": f'{stats["count"]}笔，均亏¥{stats["avg_pnl"]:,.0f}',
            })
    for bucket, stats in score_stats.items():
        if stats["count"] >= 3 and stats["win_rate"] < 0.4:
            flags.append({
                "type": "entry_score",
                "pattern": f"入场分{bucket}胜率仅{stats['win_rate']:.0%}",
                "detail": f'{stats["count"]}笔，均亏¥{stats["avg_pnl"]:,.0f}',
            })

    return ReflectionReport(
        total_trades=len(trades),
        win_rate=round(wr, 3),
        avg_pnl=round(avg_pnl, 0),
        total_pnl=round(total_pnl, 0),
        exit_reason_stats=exit_stats,
        holding_stats=holding_stats,
        score_stats=score_stats,
        flags=flags,
    )
