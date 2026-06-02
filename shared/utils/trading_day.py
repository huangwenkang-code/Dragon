"""A-share trading day utilities.

Determines last trading day accounting for weekends and Chinese holidays.
Uses AKShare's trading calendar when available, with weekend fallback.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Cache for trading calendar to avoid repeated API calls
_trading_days_cache: set[str] | None = None


def _load_trading_calendar() -> set[str]:
    """Load A-share trading days from AKShare, cached in memory."""
    global _trading_days_cache
    if _trading_days_cache is not None:
        return _trading_days_cache

    days: set[str] = set()
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is not None and not df.empty:
            col = df.columns[0]
            for d in df[col]:
                days.add(str(d)[:10])
            logger.info("Loaded %d trading days from AKShare", len(days))
    except Exception as exc:
        logger.warning("Failed to load trading calendar from AKShare: %s — using weekend-only fallback", exc)

    _trading_days_cache = days
    return days


# Hardcoded A-share holidays (fallback when AKShare unavailable).
# Market closed on these dates. Covers 2025–2026 major holidays.
_HARDCODED_HOLIDAYS: set[str] = {
    # 2025
    "2025-01-01",  # New Year
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",  # Spring Festival
    "2025-04-04", "2025-04-05", "2025-04-06",  # Qingming
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",  # Labor Day
    "2025-05-31", "2025-06-01", "2025-06-02",  # Dragon Boat
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
    "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",  # National Day
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-03",  # New Year
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-21", "2026-02-22", "2026-02-23",  # Spring Festival
    "2026-04-05", "2026-04-06", "2026-04-07",  # Qingming
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",  # Labor Day
    "2026-06-19", "2026-06-20", "2026-06-21",  # Dragon Boat
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04",
    "2026-10-05", "2026-10-06", "2026-10-07",  # National Day
}


def is_trading_day(d: date | str) -> bool:
    """Check if a date is an A-share trading day."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])

    # Weekend check (always non-trading)
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Check against hardcoded holiday list
    if d.isoformat() in _HARDCODED_HOLIDAYS:
        return False

    # Check against AKShare calendar if available
    calendar = _load_trading_calendar()
    if calendar:
        return d.isoformat() in calendar

    # Fallback: weekday = trading day
    return True


def get_last_trading_day(before: date | str | None = None) -> date:
    """Get the most recent trading day.

    Args:
        before: Reference date (defaults to today).
                On a trading day during market hours, returns today.
                On weekend/holiday, returns the previous trading day.

    Returns:
        date object representing the last trading day.
    """
    if before is None:
        ref = date.today()
    elif isinstance(before, str):
        ref = date.fromisoformat(before[:10])
    else:
        ref = before

    # Walk backwards until we find a trading day
    d = ref
    # Cap at 10 days back to avoid infinite loop
    for _ in range(10):
        if is_trading_day(d):
            return d
        d = d - timedelta(days=1)

    # Ultimate fallback: return last weekday
    logger.warning("Could not determine trading day, falling back to last weekday")
    return _last_weekday(ref)


def _last_weekday(ref: date) -> date:
    """Return the most recent weekday <= ref."""
    d = ref
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def get_trading_date_str(before: date | str | None = None) -> str:
    """Get last trading day as YYYY-MM-DD string."""
    return get_last_trading_day(before).isoformat()


def get_trading_date_compact(before: date | str | None = None) -> str:
    """Get last trading day as YYYYMMDD string."""
    return get_last_trading_day(before).strftime("%Y%m%d")


def clear_cache():
    """Clear the trading calendar cache (for testing)."""
    global _trading_days_cache
    _trading_days_cache = None
