"""Load regime config from DB with memory caching.

Usage:
    from shared.config_loader import get_config, reload_config
    cfg = await get_config("BULL")
    print(cfg["max_positions"])
"""

from __future__ import annotations

import json
import time

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_cache: dict[str, dict] = {}
_cache_time: float = 0
CACHE_TTL = 600  # 10 minutes


async def _load_from_db() -> dict[str, dict]:
    """Load all active regime configs from DB."""
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    async with async_session_factory() as s:
        r = await s.execute(sa_text(
            "SELECT regime, params FROM regime_config WHERE is_active = TRUE"
        ))
        rows = r.fetchall()
        if not rows:
            logger.warning("[config_loader] no active config in DB, using defaults")
            return {}
        return {row[0]: row[1] if isinstance(row[1], dict) else json.loads(row[1])
                for row in rows}


async def reload_config():
    """Force reload config from DB (called after optimization)."""
    global _cache, _cache_time
    _cache = await _load_from_db()
    _cache_time = time.time()
    logger.info("[config_loader] reloaded: %d regimes", len(_cache))


async def get_config(regime: str) -> dict:
    """Get config for a regime. Falls back to defaults if not in DB."""
    global _cache, _cache_time
    now = time.time()
    if not _cache or (now - _cache_time) > CACHE_TTL:
        await reload_config()

    cfg = _cache.get(regime)
    if cfg:
        return cfg

    # Fallback defaults (should never happen if migration ran)
    defaults = {
        "BULL": {"max_positions": 10, "position_size_pct": 0.20, "capital_multiplier": 1.0, "time_stop_days": 40, "filter_price_min": 1.0, "filter_amount_min": 30000000},
        "CHOPPY_UP": {"max_positions": 8, "position_size_pct": 0.18, "capital_multiplier": 0.90, "time_stop_days": 40, "filter_price_min": 1.0, "filter_amount_min": 30000000},
        "CHOPPY": {"max_positions": 6, "position_size_pct": 0.12, "capital_multiplier": 0.70, "time_stop_days": 40, "filter_price_min": 3.0, "filter_amount_min": 50000000},
        "CHOPPY_DOWN": {"max_positions": 5, "position_size_pct": 0.10, "capital_multiplier": 0.50, "time_stop_days": 40, "filter_price_min": 3.0, "filter_amount_min": 50000000},
        "BEAR": {"max_positions": 5, "position_size_pct": 0.10, "capital_multiplier": 0.50, "time_stop_days": 40, "filter_price_min": 3.0, "filter_amount_min": 50000000},
    }
    return defaults.get(regime, defaults["CHOPPY"])
