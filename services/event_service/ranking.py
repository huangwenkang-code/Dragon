"""Event ranking — sort by heat, strength, and recency."""

from __future__ import annotations

from shared.schemas.agent_state import Event
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def rank_events(
    events: list[Event],
    *,
    top_n: int = 20,
) -> list[Event]:
    """Rank events by composite score: strength × novelty × scope_factor.

    Returns top_n events, sorted descending.
    """
    scope_weights = {
        "market": 1.0,
        "sector": 0.8,
        "individual": 0.5,
    }

    scored = []
    for evt in events:
        scope_w = scope_weights.get(evt.scope, 0.5)
        composite = (
            evt.strength * 0.5
            + evt.novelty * 0.2
            + scope_w * 0.3
        )
        evt.heat_score = round(min(1.0, composite), 3)
        scored.append(evt)

    scored.sort(key=lambda e: (e.heat_score, e.strength), reverse=True)
    return scored[:top_n]


def rank_by_time(events: list[Event]) -> list[Event]:
    """Sort events by publish_time descending (most recent first)."""
    return sorted(events, key=lambda e: e.publish_time, reverse=True)


def rank_by_strength(events: list[Event]) -> list[Event]:
    """Sort events by event_strength descending."""
    return sorted(events, key=lambda e: e.event_strength, reverse=True)
