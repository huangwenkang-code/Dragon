"""Cognitive layer prompt templates."""

from services.cognitive_layer.prompts.event_extraction import event_extraction_prompt
from services.cognitive_layer.prompts.sentiment_analysis import sentiment_analysis_prompt
from services.cognitive_layer.prompts.domain_knowledge import COGNITIVE_DOMAIN_KNOWLEDGE

__all__ = [
    "event_extraction_prompt",
    "sentiment_analysis_prompt",
    "COGNITIVE_DOMAIN_KNOWLEDGE",
]
