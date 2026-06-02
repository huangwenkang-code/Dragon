"""Minimal dataflow utilities stub."""

from datetime import date, timedelta


def get_trading_date_range(base_date: str, days_before: int = 60):
    """Return (start_date, end_date) strings."""
    try:
        end_dt = date.fromisoformat(base_date)
    except (ValueError, TypeError):
        end_dt = date.today()
    start_dt = end_dt - timedelta(days=days_before)
    return start_dt.isoformat(), end_dt.isoformat()
