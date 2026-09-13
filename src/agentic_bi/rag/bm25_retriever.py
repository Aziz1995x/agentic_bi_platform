"""BM25 (lexical) retriever and dense+BM25 hybrid via EnsembleRetriever.

BM25Retriever operates on raw chunk text -- no embeddings, no vectorstore.
It builds its own in-memory term-frequency index at construction time, so
unlike load_vectorstore() it can't be persisted/loaded the same way; it's
rebuilt from your chunks each time this module is used. Fine at this
corpus size (a handful of documents); at real production scale you'd
persist this too (e.g. via a proper search engine like Elasticsearch/OpenSearch
rather than the pure-Python rank_bm25 library this wraps).
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def build_bm25_retriever(chunks: list[Document], k: int = 4) -> BM25Retriever:
    """Pure lexical retriever -- no embeddings involved at all."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def build_hybrid_retriever(
    vectorstore: VectorStore,
    chunks: list[Document],
    k: int = 4,
    dense_weight: float = 0.5,
) -> BaseRetriever:
    """Combines dense (embedding) and BM25 (lexical) retrieval via
    EnsembleRetriever, which merges both ranked lists using Reciprocal
    Rank Fusion (RRF) -- a rank-based combination, not a raw score
    average, which matters because BM25 scores and cosine-similarity
    scores live on completely different, non-comparable scales.

    dense_weight: 0.5 = equal trust in both retrievers. Push toward 1.0
    to favor semantic matches, toward 0.0 to favor exact keyword matches.
    """
    bm25_retriever = build_bm25_retriever(chunks, k=k)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, 1 - dense_weight],
    )
