"""Provider-agnostic LLM client factory.

Keeping the provider configurable (rather than hardcoding ChatAnthropic
everywhere) means swapping models for cost/latency experiments in Phase 21
(regression testing) doesn't require touching every chain — just settings.
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_bi.config.settings import get_settings


@lru_cache
def get_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a cached chat model instance based on current settings.

    lru_cache here mirrors the get_settings() pattern from Phase 0/2:
    avoids re-instantiating a client (and re-reading env vars) on every
    call, while still failing loudly at first-use rather than at import
    time if credentials are missing.
    """
    settings = get_settings()
    temp = settings.llm_temperature if temperature is None else temperature

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.anthropic_api_key,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.openai_api_key,
        )

    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.groq_api_key,
        )

    elif settings.llm_provider == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=temp,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider!r}")
