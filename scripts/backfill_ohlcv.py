"""Backfill OHLCV data from mootdx (通达信 TCP) into stock_daily_bars_v2.

Merged from backfill_ohlcv.py + fetch_all_daily_bars.py.
Only data source: mootdx TCP 7709 — no rate limit, no auth, stable.

Detection: finds stocks missing bars for recent dates (not just missing entirely).
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mootdx.quotes import Quotes
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import async_session_factory
from db.models import StockDailyBar
from shared.utils.logging import get_logger

logger = get_logger(__name__)

FETCH_BARS = 100       # bars per stock (covers ~5 months)
BATCH_SIZE = 500       # rows per commit
SLEEP_SEC = 0.10       # be nice to mootdx server


async def get_missing(session, target_dates: list[str], candidates_only: bool) -> list[str]:
    """Find stocks missing bars on ANY of the target dates.

    If candidates_only=True, only check stocks that appear in leader_candidates.
    Otherwise check all stocks in stock_daily_bars_v2.
    """
    date_list = ", ".join(f"'{d}'::date" for d in target_dates)

    if candidates_only:
        source = "(SELECT DISTINCT stock_code AS symbol FROM leader_candidates)"
    else:
        source = "(SELECT DISTINCT symbol FROM stock_daily_bars_v2)"

    sql = f"""
        SELECT s.symbol FROM {source} s
        WHERE s.symbol NOT IN (
            SELECT DISTINCT symbol FROM stock_daily_bars_v2
            WHERE trade_date IN ({date_list}) AND open != close
        )
        ORDER BY s.symbol
    """
    r = await session.execute(text(sql))
    return [row[0] for row in r.fetchall()]


def fetch_bars(client: Quotes, symbol: str, min_date: date, max_date: date) -> list[dict]:
    """Fetch bars via mootdx. Returns list of {symbol, trade_date, open, high, low, close, volume, amount}."""
    try:
        df = client.bars(symbol=symbol, frequency=9, start=0, offset=FETCH_BARS)
    except Exception as e:
        logger.warning("[%s] mootdx error: %s", symbol, e)
        return []

    if df is None or df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        dt = row.get("datetime")
        if dt is None:
            continue
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        td = dt.date() if hasattr(dt, 'date') else date.fromisoformat(str(dt)[:10])

        if td < min_date or td > max_date:
            continue

        o = float(row.get("open", 0) or 0)
        h = float(row.get("high", 0) or 0)
        l = float(row.get("low", 0) or 0)
        c = float(row.get("close", 0) or 0)
        v = float(row.get("volume", 0) or 0)
        amt = float(row.get("amount", 0) or 0)

        if all(x == 0 for x in [o, h, l, c]):
            continue

        rows.append(dict(symbol=symbol, trade_date=td, open=o, high=h, low=l, close=c, volume=v, amount=amt,
                         change_pct=0.0, turnover_pct=0.0))
    return rows


async def upsert(session, rows: list[dict]) -> int:
    if not rows:
        return 0
    seen = set()
    unique = []
    for r in rows:
        k = (r["symbol"], r["trade_date"])
        if k not in seen:
            seen.add(k)
            unique.append(r)

    stmt = pg_insert(StockDailyBar).values(unique)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "trade_date"],
        set_={"open": stmt.excluded.open, "high": stmt.excluded.high,
              "low": stmt.excluded.low, "close": stmt.excluded.close,
              "volume": stmt.excluded.volume, "amount": stmt.excluded.amount,
              "change_pct": stmt.excluded.change_pct, "turnover_pct": stmt.excluded.turnover_pct})
    await session.execute(stmt)
    return len(unique)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill OHLCV bars via mootdx")
    parser.add_argument("--days", type=int, default=7, help="Check last N days for gaps (default: 7)")
    parser.add_argument("--all", action="store_true", help="Fill all stocks (default: candidates only)")
    args = parser.parse_args()

    today = date.today()
    target_dates = [(today - timedelta(days=i)).isoformat() for i in range(1, args.days + 1)]

    logger.info("Target dates: %s", target_dates)
    logger.info("Mode: %s", "all stocks" if args.all else "candidates only")

    async with async_session_factory() as session:
        symbols = await get_missing(session, target_dates, candidates_only=not args.all)

    if not symbols:
        logger.info("All stocks up to date — nothing to do.")
        return

    logger.info("Missing: %d symbols", len(symbols))

    client = Quotes.factory(market="std", timeout=15)
    min_date = date.fromisoformat(target_dates[-1]) - timedelta(days=5)
    max_date = date.fromisoformat(target_dates[0]) + timedelta(days=1)

    inserted = 0
    failed = 0
    pending = []

    for i, sym in enumerate(symbols):
        rows = fetch_bars(client, sym, min_date, max_date)
        if rows:
            pending.extend(rows)
        elif rows is not None:
            pass
        else:
            failed += 1

        await asyncio.sleep(SLEEP_SEC)

        if len(pending) >= BATCH_SIZE or i == len(symbols) - 1:
            if pending:
                async with async_session_factory() as s:
                    async with s.begin():
                        n = await upsert(s, pending)
                        inserted += n
                pending = []

        if (i + 1) % 50 == 0 or i == len(symbols) - 1:
            logger.info("[%d/%d] %d upserted, %d failed", i + 1, len(symbols), inserted, failed)

    logger.info("Done. %d symbols, %d rows upserted, %d failed", len(symbols), inserted, failed)


if __name__ == "__main__":
    asyncio.run(main())
