"""Unified market regime detection — BULL / CHOPPY_UP / CHOPPY / CHOPPY_DOWN / BEAR.

Uses 上证指数 (000001.SH) MA20 + slope.
"""

from __future__ import annotations

from datetime import date as dt_date, timedelta

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_cache: dict[str, str] = {}


async def _query_closes(trade_date: str) -> list[float]:
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    td = dt_date.fromisoformat(trade_date[:10])
    lb = td - timedelta(days=60)
    async with async_session_factory() as session:
        r = await session.execute(
            sa_text(
                "SELECT close FROM stock_daily_bars_v2 "
                "WHERE symbol='000001.SH' AND trade_date >= :lb AND trade_date <= :td "
                "ORDER BY trade_date"
            ),
            {"lb": lb, "td": td},
        )
        return [float(row[0] or 0) for row in r.fetchall() if row[0]]


async def detect_regime(trade_date: str) -> str:
    key = trade_date[:10]
    if key in _cache:
        return _cache[key]

    try:
        closes = await _query_closes(trade_date)
    except Exception:
        _cache[key] = "CHOPPY"
        return "CHOPPY"

    if len(closes) < 26:
        _cache[key] = "CHOPPY"
        return "CHOPPY"

    today_close = closes[-1]
    ma20 = sum(closes[-21:-1]) / 20
    ma20_5d = sum(closes[-26:-6]) / 20

    pma = (today_close - ma20) / ma20 if ma20 > 0 else 0
    slope = (ma20 - ma20_5d) / ma20_5d if ma20_5d > 0 else 0

    if pma > 0.02 and slope > 0.005:
        regime = "BULL"
    elif pma < -0.02 and slope < -0.005:
        regime = "BEAR"
    elif slope > 0.002:
        regime = "CHOPPY_UP"
    elif slope < -0.002:
        regime = "CHOPPY_DOWN"
    else:
        regime = "CHOPPY"

    _cache[key] = regime
    logger.debug("[market_regime] %s → %s (pma=%.3f slope=%.3f)", key, regime, pma, slope)
    return regime


def clear_cache():
    _cache.clear()
