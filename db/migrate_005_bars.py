"""Migration 005: Create stock_daily_bars table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from db.connection import engine
from db.models import Base
from sqlalchemy import text

async def migrate():
    async with engine.begin() as conn:
        # Create the table via ORM (same pattern as migrate_backtest.py)
        await conn.run_sync(Base.metadata.create_all, tables=[
            Base.metadata.tables["stock_daily_bars"],
        ])
        # Create additional indexes
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_bars_symbol ON stock_daily_bars(symbol)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_bars_date ON stock_daily_bars(trade_date)"
        ))
    print("[005] stock_daily_bars table created")

if __name__ == "__main__":
    asyncio.run(migrate())
