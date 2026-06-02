"""TokenUsageTracker — singleton + LangChain callback handler."""

from __future__ import annotations

from datetime import datetime
from langchain_core.callbacks.base import BaseCallbackHandler
from services.token_tracker.models import TokenRecord
from services.token_tracker.pricing import estimate_cost
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback that captures token usage from every LLM call."""

    def __init__(self, tracker: "TokenUsageTracker"):
        self._tracker = tracker

    def on_llm_end(self, response, **kwargs):
        try:
            llm_output = response.llm_output or {}
            # Try multiple LLM provider formats
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if not usage:
                # LangChain >= 0.3 some versions: usage_metadata on response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                        "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                        "total_tokens": response.usage_metadata.get("total_tokens", 0),
                    }
                elif hasattr(response, 'response_metadata') and response.response_metadata:
                    usage = response.response_metadata.get("token_usage", {})
            if not usage:
                # Last attempt: AIMessage.usage_metadata via generations
                generations = getattr(response, 'generations', [])
                if generations and len(generations) > 0:
                    gen0 = generations[0]
                    if hasattr(gen0, 'message'):
                        msg = gen0[0] if isinstance(gen0, list) and len(gen0) > 0 else gen0
                        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                            um = msg.usage_metadata
                            usage = {
                                "prompt_tokens": um.get("input_tokens", 0),
                                "completion_tokens": um.get("output_tokens", 0),
                                "total_tokens": um.get("total_tokens", 0),
                            }
                if not usage:
                    logger.debug("No token_usage found in response. llm_output keys: %s, response type: %s",
                               list(llm_output.keys()), type(response).__name__)
                    return
            model = llm_output.get("model_name", "") or getattr(response, 'model_name', "") or "unknown"
            record = TokenRecord(
                run_id=self._tracker.current_run_id,
                step=kwargs.get("name", "unknown"),
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cost=estimate_cost(model,
                    usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                    usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)),
                timestamp=datetime.now().isoformat(),
            )
            self._tracker.record(record)
        except Exception as e:
            logger.debug("Token tracking callback error: %s", e)


class TokenUsageTracker:
    """Singleton tracking token usage across a pipeline run."""

    _instance: TokenUsageTracker | None = None

    def __init__(self):
        self.current_run_id: str = ""
        self.records: list[TokenRecord] = []

    @classmethod
    def instance(cls) -> TokenUsageTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_run(self, run_id: str):
        self.current_run_id = run_id
        self.records = []

    def record(self, record: TokenRecord):
        if record.run_id == self.current_run_id or not record.run_id:
            record.run_id = self.current_run_id
        self.records.append(record)

    def summary(self) -> dict:
        total_input = sum(r.prompt_tokens for r in self.records)
        total_output = sum(r.completion_tokens for r in self.records)
        total_cost = sum(r.cost for r in self.records)
        return {
            "run_id": self.current_run_id,
            "total_prompt_tokens": total_input,
            "total_completion_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(total_cost, 6),
            "records": [r.to_dict() for r in self.records],
        }

    def create_callback(self) -> TokenTrackingCallback:
        return TokenTrackingCallback(self)
