"""Backfill Shanghai Composite Index (000001.SH) data into stock_daily_bars."""
import asyncio
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mootdx.quotes import Quotes
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from db.connection import async_session_factory
from db.models import StockDailyBar

INDEX_TDX_SYMBOL = "999999"
INDEX_DB_SYMBOL = "000001.SH"


async def main():
    print("Fetching Shanghai Composite Index from mootdx...")
    client = Quotes.factory(market="std", timeout=15)
    df = client.index(symbol=INDEX_TDX_SYMBOL, frequency=9, start=0, offset=300)

    if df is None or df.empty:
        print("ERROR: No index data returned")
        return

    print(f"Got {len(df)} index bars, {df.index[0]} to {df.index[-1]}")

    rows = []
    for idx, row in df.iterrows():
        dt = idx.to_pydatetime().date() if hasattr(idx, 'to_pydatetime') else idx.date()
        rows.append(dict(
            symbol=INDEX_DB_SYMBOL,
            trade_date=dt,
            open=float(row.get("open", 0) or 0),
            high=float(row.get("high", 0) or 0),
            low=float(row.get("low", 0) or 0),
            close=float(row.get("close", 0) or 0),
            volume=float(row.get("volume", 0) or 0),
            amount=float(row.get("amount", 0) or 0),
            change_pct=0.0,
            turnover_pct=0.0,
        ))

    # Deduplicate
    seen = set()
    unique = []
    for r in rows:
        key = (r["symbol"], r["trade_date"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Upserting {len(unique)} unique rows...")

    async with async_session_factory() as session:
        async with session.begin():
            stmt = pg_insert(StockDailyBar).values(unique)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "trade_date"],
                set_={
                    "open": stmt.excluded.open, "high": stmt.excluded.high,
                    "low": stmt.excluded.low, "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume, "amount": stmt.excluded.amount,
                    "change_pct": stmt.excluded.change_pct,
                    "turnover_pct": stmt.excluded.turnover_pct,
                }
            )
            await session.execute(stmt)

    # Verify
    async with async_session_factory() as session:
        r = await session.execute(
            text(f"SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM stock_daily_bars WHERE symbol = '{INDEX_DB_SYMBOL}'")
        )
        cnt, mn, mx = r.fetchone()
        print(f"Done. {INDEX_DB_SYMBOL}: {cnt} rows, {mn} to {mx}")


if __name__ == "__main__":
    asyncio.run(main())
