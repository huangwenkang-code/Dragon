"""llm_provider — canonical LLM access point.

Wraps the copied TradingAgents-CN llm_clients / llm_adapters.
All service code calls `create_llm()` from here.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from shared.configs.settings import get_settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Create a LangChain-compatible chat model.

    Delegates to the TradingAgents-CN factory for provider resolution,
    with fallback to direct OpenAI-compatible construction.
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    # Attempt to use the TradingAgents-CN llm_clients factory
    try:
        from services.llm_adapter.llm_clients.factory import create_llm_client

        raw = create_llm_client(
            provider=provider,
            model=model or settings.quick_think_llm,
            base_url=settings.llm_backend_url or None,
            api_key=settings.llm_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm = raw.get_llm()
        logger.info("create_llm: using TradingAgents-CN factory → %s/%s", provider, model)
    except Exception as exc:
        logger.debug("TradingAgents-CN llm factory unavailable: %s", exc)
        # Fallback to direct construction
        llm = _fallback_llm(provider, model, temperature, max_tokens)

    # Inject token tracking callback so every LLM invocation is measured
    try:
        from services.token_tracker import TokenUsageTracker
        tracker = TokenUsageTracker.instance()
        callback = tracker.create_callback()
        if hasattr(llm, 'callbacks'):
            llm.callbacks = (llm.callbacks or []) + [callback]
        else:
            llm.callbacks = [callback]
    except Exception:
        pass

    return llm


def create_deep_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    settings = get_settings()
    return create_llm(
        provider=provider,
        model=model or settings.deep_think_llm,
        temperature=0.0,
        max_tokens=8192,
    )


def create_quick_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    settings = get_settings()
    return create_llm(
        provider=provider,
        model=model or settings.quick_think_llm,
        temperature=0.0,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_llm(
    provider: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    settings = get_settings()
    provider_lower = provider.lower()

    if provider_lower in ("openai", "deepseek", "qwen", "glm", "qianfan", "siliconflow", "openrouter", "aihubmix", "ollama"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.quick_think_llm,
            base_url=settings.llm_backend_url or "https://api.openai.com/v1",
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider_lower == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-sonnet-4-6",
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider_lower == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=settings.llm_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
