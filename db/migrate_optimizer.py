"""Create regime_config table + insert V1 initial params."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import async_session_factory
from sqlalchemy import text as sa_text

INITIAL_PARAMS = {
    "BULL": {
        "max_positions": 10, "position_size_pct": 0.20,
        "capital_multiplier": 1.0, "time_stop_days": 40,
        "filter_price_min": 1.0, "filter_amount_min": 30000000,
        "selection": "momentum",
    },
    "CHOPPY_UP": {
        "max_positions": 8, "position_size_pct": 0.18,
        "capital_multiplier": 0.90, "time_stop_days": 40,
        "filter_price_min": 1.0, "filter_amount_min": 30000000,
        "selection": "momentum",
    },
    "CHOPPY": {
        "max_positions": 6, "position_size_pct": 0.12,
        "capital_multiplier": 0.70, "time_stop_days": 40,
        "filter_price_min": 3.0, "filter_amount_min": 50000000,
        "selection": "blend",
    },
    "CHOPPY_DOWN": {
        "max_positions": 5, "position_size_pct": 0.10,
        "capital_multiplier": 0.50, "time_stop_days": 40,
        "filter_price_min": 3.0, "filter_amount_min": 50000000,
        "selection": "oversold_bounce",
    },
    "BEAR": {
        "max_positions": 5, "position_size_pct": 0.10,
        "capital_multiplier": 0.50, "time_stop_days": 40,
        "filter_price_min": 3.0, "filter_amount_min": 50000000,
        "selection": "oversold_bounce",
    },
}


async def migrate():
    async with async_session_factory() as s:
        # Create table
        await s.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS regime_config (
                id SERIAL PRIMARY KEY,
                regime VARCHAR(20) NOT NULL,
                params JSONB NOT NULL DEFAULT '{}',
                score DOUBLE PRECISION DEFAULT 0,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(regime, is_active)
            )
        """))
        # Partial unique index: only one active row per regime
        await s.execute(sa_text("""
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_regime
            ON regime_config (regime) WHERE is_active = TRUE
        """))
        await s.commit()
        print("Table regime_config created.")

        # Check if data exists
        r = await s.execute(sa_text("SELECT COUNT(*) FROM regime_config"))
        if r.scalar() > 0:
            print(f"Already has {r.scalar()} rows, skipping init.")
            return

        # Insert V1 initial params
        for regime, params in INITIAL_PARAMS.items():
            await s.execute(sa_text(
                "INSERT INTO regime_config (regime, params, is_active, score) "
                "VALUES (:r, :p, TRUE, 0)"
            ), {"r": regime, "p": json.dumps(params)})
        await s.commit()
        print(f"Inserted {len(INITIAL_PARAMS)} initial configs.")


if __name__ == "__main__":
    asyncio.run(migrate())
