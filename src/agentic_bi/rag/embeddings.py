"""Embedding model factory.

Note: Groq does not offer an embeddings API (confirmed as of this writing —
they serve fast open-model chat inference and Whisper STT only). If you want
Groq for the *chat* model (fast Llama inference), that's a separate addition
to llm/client.py, not this file.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from agentic_bi.config.settings import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    elif settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        # Runs fully offline on CPU -- no API key, no per-call cost, no
        # network dependency. Slower per-call and lower retrieval quality
        # than text-embedding-3-small on most benchmarks, but useful for:
        # (a) fast local iteration without burning API credits, and
        # (b) Phase 21 regression runs where you re-embed the eval set
        #     repeatedly and don't want that cost to compound.
        # First call downloads the model (~90MB) and caches it locally.
        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

    raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider!r}")
