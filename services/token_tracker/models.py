"""Token tracking data models."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenRecord:
    """A single LLM call's token usage."""
    run_id: str = ""
    step: str = ""               # e.g. "event_extraction", "sentiment_enrichment"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp,
        }
