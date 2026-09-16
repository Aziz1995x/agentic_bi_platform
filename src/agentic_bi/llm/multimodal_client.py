"""Provider-agnostic multimodal LLM client factory.

Supports multimodal models from OpenAI and Ollama.
The provider and model are configurable through application settings,
allowing vision-capable models to be swapped without changing chains.
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_bi.config.settings import get_settings


@lru_cache
def get_multimodal_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a cached multimodal chat model based on current settings."""

    settings = get_settings()
    temp = settings.llm_temperature if temperature is None else temperature

    if settings.multimodal_llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.multimodal_llm_model,
            temperature=temp,
            api_key=settings.openai_api_key,
        )

    elif settings.multimodal_llm_provider == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.multimodal_llm_model,
            temperature=temp,
            api_key=settings.ollama_api_key,
            base_url=settings.ollama_cloud_base_url,
        )

    raise ValueError(f"Unsupported multimodal llm_provider: {settings.multimodal_llm_model!r}")