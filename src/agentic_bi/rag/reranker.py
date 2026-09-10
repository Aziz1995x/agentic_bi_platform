"""Reranker factory + retriever wrapper.

A reranker doesn't replace the base retriever -- it sits after it. The
pattern is always: base retriever fetches a WIDE candidate pool (fetch_k,
cheap vector search), then the reranker re-scores each candidate against
the exact query text (expensive per-candidate, but only run on the small
fetched pool, not the whole index) and returns the top_n after re-sorting.

This is the same "cheap wide net, then reranked to a precise dozen" pattern
you saw in MultiQuery, but the re-scoring step here is a cross-encoder --
it looks at (query, chunk) pairs jointly, rather than comparing two
independently-computed embedding vectors. That joint view is why
cross-encoders are typically more accurate than embedding similarity for
relevance judgments, and also why they're too slow to run against an
entire index -- fine for re-scoring 10-20 candidates, not fine for 30,000.
"""

from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from agentic_bi.config.settings import get_settings


@lru_cache
def get_reranker(top_n: int = 4) -> BaseDocumentCompressor:
    settings = get_settings()

    if settings.reranker_provider == "cohere":
        from langchain_cohere import CohereRerank

        # Free trial key: 1,000 calls/month, 10 req/min on rerank
        # specifically -- fine for this eval set, not for production load.
        return CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model=settings.cohere_model,
            top_n=top_n,
        )

    elif settings.reranker_provider == "local":
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        # Runs fully offline on CPU. Slower per-query than Cohere's hosted
        # API for a single request, but no rate limit and no API cost --
        # matters once you're running the full eval set repeatedly (Phase
        # 21 regression testing), same tradeoff as local vs. OpenAI
        # embeddings from Phase 4.
        cross_encoder = HuggingFaceCrossEncoder(model_name=settings.local_reranker_model)
        return CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    raise ValueError(f"Unsupported reranker_provider: {settings.reranker_provider!r}")


def get_reranking_retriever(
    vectorstore: VectorStore,
    fetch_k: int = 10,
    top_n: int = 4,
) -> BaseRetriever:
    """Wraps a base retriever with a reranking compression step.

    fetch_k: candidates pulled by cheap vector search before reranking.
    top_n: final count returned after the cross-encoder re-scores them.

    fetch_k should be meaningfully larger than top_n -- if they're equal,
    reranking can only reorder what plain similarity search already found,
    which defeats the purpose (case #8's whole premise is that the right
    chunk sits at rank 3 in the top-4; reranking needs a wider net than 4
    to have any chance of promoting a chunk that similarity search ranked,
    say, 6th or 7th but was never even considered).
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    compressor = get_reranker(top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
