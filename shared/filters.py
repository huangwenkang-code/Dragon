"""Hard quality filters for candidate stock pre-screening.

Reduces noise from 3000+ → 200-500 candidates before scoring.

Filters (all must pass):
  1. NOT ST or *ST
  2. NOT consecutive 2-year loss (if data available)
  3. Daily turnover >= 50M yuan (liquidity floor)
  4. Price >= 3 yuan (penny stock exclusion)
  5. Market cap >= 2B yuan (shell stock exclusion)
  6. NOT limit-down today (< -9.8%)
"""

from __future__ import annotations

import asyncio
from datetime import date as dt_date, timedelta

from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def fetch_stock_basics(symbols: list[str]) -> dict[str, dict]:
    """Fetch stock_basics for a list of symbols.

    Returns {symbol: {name, industry, market_cap, pe, pb, list_date}}
    """
    if not symbols:
        return {}

    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    async with async_session_factory() as session:
        r = await session.execute(
            sa_text(
                "SELECT stock_code, stock_name, industry, market_cap, pe, pb, list_date "
                "FROM stock_basics WHERE stock_code = ANY(:syms)"
            ),
            {"syms": symbols},
        )
        rows = r.fetchall()
        return {
            row[0]: {
                "stock_name": row[1] or "",
                "industry": row[2] or "",
                "market_cap": float(row[3] or 0),
                "pe": float(row[4] or 0),
                "pb": float(row[5] or 0),
                "list_date": row[6] or "",
            }
            for row in rows
        }


async def fetch_daily_data(
    symbols: list[str], trade_date: str
) -> dict[str, dict]:
    """Fetch today's OHLCV + turnover + change_pct for a list of symbols.

    Returns {symbol: {close, change_pct, turnover_pct, amount, volume}}
    """
    if not symbols:
        return {}

    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    td = dt_date.fromisoformat(trade_date[:10])
    async with async_session_factory() as session:
        r = await session.execute(
            sa_text(
                "SELECT symbol, close, change_pct, turnover_pct, amount, volume "
                "FROM stock_daily_bars_v2 "
                "WHERE symbol = ANY(:syms) AND trade_date = :td"
            ),
            {"syms": symbols, "td": td},
        )
        rows = r.fetchall()
        return {
            row[0]: {
                "close": float(row[1] or 0),
                "change_pct": float(row[2] or 0),
                "turnover_pct": float(row[3] or 0),
                "amount": float(row[4] or 0),
                "volume": float(row[5] or 0),
            }
            for row in rows
        }


def apply_filters(
    candidates: list[dict],
    basics: dict[str, dict],
    daily_data: dict[str, dict],
    *,
    trade_date: str = "",
) -> tuple[list[dict], list[dict]]:
    """Apply hard quality filters. Returns (passed, rejected).

    Each candidate dict must have at least 'symbol' and 'stock_code' keys.
    """
    passed: list[dict] = []
    rejected: list[dict] = []

    for c in candidates:
        sym = c.get("symbol", "") or c.get("stock_code", "")
        if not sym:
            rejected.append({**c, "_reject_reason": "no symbol"})
            continue

        basic = basics.get(sym, {})
        daily = daily_data.get(sym, {})

        # ---- 1. ST filter (skip if no stock_basics data) ----
        name = basic.get("stock_name", "") or c.get("stock_name", "")
        if name and ("ST" in name.upper() or "*ST" in name.upper()):
            rejected.append({**c, "_reject_reason": "ST"})
            continue

        # ---- 2. Penny stock (< 3 yuan) ----
        price = daily.get("close", 0) or c.get("price", 0)
        if 0 < price < 3.0:
            rejected.append({**c, "_reject_reason": f"penny stock ({price:.2f})"})
            continue

        # ---- 3. Liquidity floor (< 50M yuan turnover) ----
        amount = daily.get("amount", 0) or c.get("amount", 0) or 0
        if 0 < amount < 50_000_000:  # 5000万
            rejected.append({**c, "_reject_reason": f"low liquidity ({amount/1e4:.0f}万)"})
            continue

        # ---- 4. Shell stock (< 2B market cap) — skip if no data ----
        mcap = basic.get("market_cap", 0) or c.get("market_cap", 0) or 0
        if mcap > 0 and mcap < 2_000_000_000:  # only filter if we have the data
            rejected.append({**c, "_reject_reason": f"shell stock ({mcap/1e8:.1f}亿)"})
            continue

        # ---- 5. Limit down today ----
        change_pct = daily.get("change_pct", 0) or c.get("change_pct", 0) or 0
        if change_pct <= -9.8:
            rejected.append({**c, "_reject_reason": "limit down"})
            continue

        passed.append(c)

    return passed, rejected


# ---------------------------------------------------------------------------
# BEAR-specific: oversold bounce candidates
# ---------------------------------------------------------------------------

async def find_oversold_bounce(
    trade_date: str, top_n: int = 60
) -> list[dict]:
    """BEAR mode: find quality stocks that are oversold (>30% off 60d high).

    Filters applied in SQL (no stock_basics dependency):
      - price >= 3
      - 60d high drawdown > 30%
      - daily turnover > 5000万
      - exclude symbols with 'ST' in the data (via name check if available)
    """
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    td = dt_date.fromisoformat(trade_date[:10])
    lookback_60d = td - timedelta(days=80)  # extra buffer for holidays

    async with async_session_factory() as session:
        r = await session.execute(
            sa_text("""
                WITH today_bars AS (
                    SELECT symbol, close, change_pct, turnover_pct, amount, volume
                    FROM stock_daily_bars_v2
                    WHERE trade_date = :td AND close > 0 AND volume > 0
                ),
                sixty_day_high AS (
                    SELECT symbol, MAX(high) as high_60d
                    FROM stock_daily_bars_v2
                    WHERE trade_date >= :lb AND trade_date <= :td
                      AND high > 0
                    GROUP BY symbol
                )
                SELECT t.symbol, t.close, t.change_pct,
                       t.turnover_pct, t.amount,
                       ROUND(CAST((s.high_60d - t.close) / s.high_60d * 100 AS numeric), 1) as drawdown_pct
                FROM today_bars t
                JOIN sixty_day_high s ON t.symbol = s.symbol
                WHERE t.close >= 3
                  AND t.amount >= 50000000  -- 5000万
                  AND (s.high_60d - t.close) / s.high_60d > 0.30  -- 距60日高点跌>30%
                ORDER BY drawdown_pct DESC, t.amount DESC
                LIMIT :top_n
            """),
            {"td": td, "lb": lookback_60d, "top_n": top_n},
        )
        rows = r.fetchall()

    results = []
    for row in rows:
        sym, close, chg, turnover, amt, dd = row
        results.append({
            "symbol": sym,
            "stock_name": sym,
            "price": float(close or 0),
            "change_pct": float(chg or 0),
            "turnover_pct": float(turnover or 0),
            "amount": float(amt or 0),
            "amount_wan": round(float(amt or 0) / 10000, 2),
            "market_cap": 0,
            "pe": 0,
            "drawdown_pct": float(dd or 0),
            "flow_score": round(float(dd or 0) / 100, 3),
            "_source": "oversold_bounce",
        })

    logger.info(
        "[filters] oversold_bounce: %d candidates for %s (top drawdown=%.1f%%)",
        len(results), trade_date, results[0]["drawdown_pct"] if results else 0,
    )
    return results


# ---------------------------------------------------------------------------
# BULL/CHOPPY_UP: concept leaders from 同花顺热点 (sector_concepts)
# ---------------------------------------------------------------------------

async def fetch_concept_leaders(
    trade_date: str, top_n: int = 40
) -> list[dict]:
    """Fetch concept leader stocks from sector_concepts table.

    These are stocks identified as 同花顺热点概念龙头 on the given date.
    Used for BULL/CHOPPY_UP modes to capture theme-driven surges.

    Returns list of candidate dicts with symbol, concept_name, leader_score.
    """
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    td = dt_date.fromisoformat(trade_date[:10])

    async with async_session_factory() as session:
        # Get distinct concept leaders, deduplicated by stock
        r = await session.execute(
            sa_text("""
                WITH ranked AS (
                    SELECT DISTINCT ON (leader_stock)
                        leader_stock, leader_stock_name, concept_name,
                        leader_stock_change, heat
                    FROM sector_concepts
                    WHERE snapshot_date = :td
                      AND leader_stock IS NOT NULL
                      AND leader_stock != ''
                    ORDER BY leader_stock, heat DESC
                )
                SELECT leader_stock, leader_stock_name, concept_name,
                       leader_stock_change, heat
                FROM ranked
                ORDER BY heat DESC, leader_stock_change DESC
                LIMIT :top_n
            """),
            {"td": td, "top_n": top_n},
        )
        rows = r.fetchall()

    if not rows:
        # Fallback: try adjacent trading days (concept data might lag by 1 day)
        async with async_session_factory() as session:
            r = await session.execute(
                sa_text("""
                    WITH ranked AS (
                        SELECT DISTINCT ON (leader_stock)
                            leader_stock, leader_stock_name, concept_name,
                            leader_stock_change, heat
                        FROM sector_concepts
                        WHERE snapshot_date IN (
                            SELECT snapshot_date FROM sector_concepts
                            WHERE snapshot_date <= :td
                            ORDER BY snapshot_date DESC LIMIT 3
                        )
                          AND leader_stock IS NOT NULL AND leader_stock != ''
                        ORDER BY leader_stock, heat DESC
                    )
                    SELECT leader_stock, leader_stock_name, concept_name,
                           leader_stock_change, heat
                    FROM ranked
                    ORDER BY heat DESC, leader_stock_change DESC
                    LIMIT :top_n
                """),
                {"td": td, "top_n": top_n},
            )
            rows = r.fetchall()

    results = []
    for row in rows:
        sym, name, concept, chg, heat = row
        # Score: concept leader gets higher base score
        score = round(min(0.70, 0.45 + abs(float(chg or 0)) / 100 + float(heat or 0) / 200), 4)
        results.append({
            "symbol": sym,
            "stock_name": name or sym,
            "concept_name": concept or "",
            "change_pct": float(chg or 0),
            "heat": int(heat or 0),
            "leader_score": score,
            "_source": "concept_leader",
        })

    logger.info(
        "[filters] concept_leaders: %d candidates for %s (top: %s %s)",
        len(results), trade_date,
        results[0]["symbol"] if results else "none",
        results[0]["concept_name"] if results else "",
    )
    return results
