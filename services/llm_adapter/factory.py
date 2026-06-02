"""LLM provider factory — adapter over TradingAgents-CN llm_clients."""

from typing import Optional

from langchain_core.language_models import BaseChatModel

from shared.configs.settings import get_settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Maps provider keys to (langchain_class, default_base_url)
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai.ChatOpenAI", "https://api.openai.com/v1"),
    "deepseek": ("langchain_openai.ChatOpenAI", "https://api.deepseek.com/v1"),
    "qwen": ("langchain_openai.ChatOpenAI", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm": ("langchain_openai.ChatOpenAI", "https://open.bigmodel.cn/api/paas/v4"),
    "ollama": ("langchain_ollama.ChatOllama", ""),
    "google": ("langchain_google_genai.ChatGoogleGenerativeAI", ""),
    "anthropic": ("langchain_anthropic.ChatAnthropic", ""),
}

_ALIASES = {
    "dashscope": "qwen",
    "alibaba": "qwen",
    "zhipu": "glm",
    "siliconflow": "openai",
}


def _resolve_provider(raw: str) -> str:
    return _ALIASES.get(raw.lower(), raw.lower())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Create a LangChain-compatible chat model for *provider*.

    Uses shared Settings for API keys and base URLs.
    """
    settings = get_settings()
    provider = _resolve_provider(provider or settings.llm_provider)

    if provider not in _PROVIDER_REGISTRY:
        logger.warning(
            "Unknown provider '%s', falling back to OpenAI-compatible", provider
        )
        provider = "openai"

    cls_path, default_url = _PROVIDER_REGISTRY[provider]
    backend_url = settings.llm_backend_url or default_url

    # OpenAI-compatible path (most providers)
    if provider in ("openai", "deepseek", "qwen", "glm"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.quick_think_llm,
            base_url=backend_url,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Anthropic
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-sonnet-4-6",
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Google
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=settings.llm_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # Ollama
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or "llama3",
            base_url=backend_url or "http://localhost:11434",
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_deep_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """Create the 'deep think' LLM (for complex reasoning)."""
    settings = get_settings()
    return create_llm(
        provider=provider,
        model=model or settings.deep_think_llm,
        temperature=0.0,
        max_tokens=8192,
    )


def create_quick_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """Create the 'quick think' LLM (for simple classification)."""
    settings = get_settings()
    return create_llm(
        provider=provider,
        model=model or settings.quick_think_llm,
        temperature=0.0,
        max_tokens=2048,
    )
