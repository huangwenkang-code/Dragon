"""Migration: add backtest tables + token_usage column to pipeline_runs."""

from db.connection import engine
from db.models import Base
from sqlalchemy import text


async def migrate():
    async with engine.begin() as conn:
        # Create new tables
        await conn.run_sync(Base.metadata.create_all, tables=[
            Base.metadata.tables["backtest_strategies"],
            Base.metadata.tables["backtest_runs"],
            Base.metadata.tables["backtest_trades"],
        ])
        # Add token_usage column if not exists (SQLite-compatible approach)
        try:
            await conn.execute(text(
                "ALTER TABLE pipeline_runs ADD COLUMN token_usage JSON DEFAULT '{}'"
            ))
        except Exception:
            pass  # column already exists (or SQLite doesn't support ADD COLUMN IF NOT EXISTS)
    print("Migration complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(migrate())
