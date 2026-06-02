"""Cognitive layer — concept-level news → event extraction → sentiment analysis.

The pipeline transforms raw sector/concept news into structured market events,
then analyzes multi-dimensional sentiment. This is the "brain" of dragon-engine.
"""

from services.cognitive_layer.news_fetcher import fetch_concept_news, extract_concept_names
from services.cognitive_layer.event_extractor import extract_events_from_news

__all__ = [
    "fetch_concept_news",
    "extract_concept_names",
    "extract_events_from_news",
]
