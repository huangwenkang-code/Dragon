"""analyze_sentiment node — calls real sentiment-service for event analysis."""

from shared.schemas.agent_state import AgentState
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def analyze_sentiment(state: AgentState) -> dict:
    """Analyze sentiment and narrative for discovered events.

    Calls sentiment-service for:
      1. Keyword-based sentiment scoring
      2. LLM-enhanced sentiment summary
      3. Narrative unity assessment
      4. Hot topic / sector detection
    """
    from services.sentiment_service.analyzer import analyze_sentiment as do_analyze

    result = await do_analyze(state)

    # Enrich with narrative engine
    try:
        from services.sentiment_service.narrative_engine import (
            assess_narrative_unity,
            assess_monster_stock_potential,
        )
        from shared.schemas.agent_state import SentimentScore

        scores = [SentimentScore(**s) for s in result.get("sentiment_scores", [])]
        narrative = assess_narrative_unity(scores)
        monster = assess_monster_stock_potential(scores, narrative)

        result["sentiment_summary"] = (
            f"{result.get('sentiment_summary', '')} | "
            f"叙事统一: {'是' if narrative['is_unified'] else '否'} "
            f"({narrative['narrative_direction']}) | "
            f"妖股潜力: {monster['potential']:.2f} | "
            f"扩散概率: {narrative['propagation_score']:.2f}"
        )
    except Exception as exc:
        logger.warning("narrative engine enrichment failed: %s", exc)

    logger.info("[analyze_sentiment] produced %d scores", len(result.get("sentiment_scores", [])))
    return result
