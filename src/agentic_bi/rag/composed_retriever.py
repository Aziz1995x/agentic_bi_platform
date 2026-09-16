"""Composed pipeline: Hybrid (Dense + BM25, RRF fusion) generates a wide
candidate pool, then a cross-encoder reranker narrows it to the final
top_n. This is the pattern Anthropic's own Contextual Retrieval research
uses in production -- complementary techniques at different stages,
tested together only after each stage was independently measured.

Stage 1 (Hybrid) must run wider than the final desired result count --
if fetch_k here equals top_n, the reranker can only reorder what Hybrid
already surfaced, defeating the purpose of giving it a wider net to
rerank from.
"""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from agentic_bi.rag.bm25_retriever import build_hybrid_retriever
from agentic_bi.rag.reranker import get_reranker


def build_composed_retriever(
    vectorstore: VectorStore,
    chunks: list[Document],
    fetch_k: int = 10,
    top_n: int = 4,
    dense_weight: float = 0.5,
) -> BaseRetriever:
    hybrid_retriever = build_hybrid_retriever(
        vectorstore, chunks, k=fetch_k, dense_weight=dense_weight
    )
    compressor = get_reranker(top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )
