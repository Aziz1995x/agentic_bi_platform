"""
retriever.py
────────────
Query-time retrieval for the multimodal FAISS index.

What this does differently from the text retriever
───────────────────────────────────────────────────
The text RAG retriever returns Documents whose page_content is the
chunk to pass into the LLM prompt. The multimodal retriever does that
too, but also returns structured result objects that carry:

  - the retrieved text (caption or PDF text)
  - the file_path of the original image/PDF (so the UI can render it)
  - source_type ("chart" | "table" | "pdf_text" | "pdf_image")
  - similarity score (FAISS cosine distance, lower = more similar)
  - all original metadata

This structure matters because "retrieve the chart" means the agent
needs to give the user back an image path, not just the caption text.

Two retrieval modes
───────────────────
1. similarity_search (default) — plain cosine similarity, same as Phase 5
   basic retriever.  This is what we evaluate first.

2. mmr_search — Maximal Marginal Relevance, same tradeoff as Phase 5:
   reduces redundancy at the cost of some precision.  Offered here as a
   parameter option so the eval harness can compare both without a code
   change.

Source-type filtering
─────────────────────
The caller can restrict results to specific source types, e.g.:
  retrieve("return rate chart", source_types=["chart", "table"])
This is metadata filtering on post-retrieved docs, not at FAISS level.
We fetch k * filter_oversample candidates then filter, so k results are
still returned when possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SourceType = Literal["chart", "table", "pdf_text", "pdf_image"]


@dataclass
class MultimodalResult:
    """
    A single retrieved artefact from the multimodal index.

    Attributes
    ----------
    content:      The embedded text (caption for images, extracted text for PDFs).
    source_type:  One of "chart", "table", "pdf_text", "pdf_image".
    file_path:    Absolute path to the original file (image or PDF).
    file_name:    Filename only, e.g. "04_category_return_rate.png".
    title:        Human-readable title from the corpus metadata.
    score:        Cosine similarity score (0–1, higher = more similar).
    metadata:     Full metadata dict from the LangChain Document.
    """
    content:     str
    source_type: str
    file_path:   str
    file_name:   str
    title:       str
    score:       float
    metadata:    dict = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.source_type in ("chart", "table", "pdf_image")

    @property
    def is_text(self) -> bool:
        return self.source_type == "pdf_text"

    def __repr__(self) -> str:
        return (
            f"MultimodalResult(type={self.source_type!r}, "
            f"file={self.file_name!r}, score={self.score:.3f})"
        )


class MultimodalRetriever:
    """
    Wraps a multimodal FAISS vectorstore with structured result objects
    and optional source-type filtering.

    Usage
    -----
        from agentic_bi.rag.multimodal.retriever import MultimodalRetriever
        from agentic_bi.rag.multimodal.ingest import load_multimodal_vectorstore

        retriever = MultimodalRetriever(load_multimodal_vectorstore())
        results = retriever.retrieve("return rate by product category", k=4)
        for r in results:
            print(r.file_name, r.score)
    """

    # When filtering by source_type, fetch this many extra candidates so
    # we still return k results after filtering removes some.
    _FILTER_OVERSAMPLE = 3

    def __init__(self, vectorstore: FAISS) -> None:
        self._vs = vectorstore

    # ── Core retrieval ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 4,
        source_types: list[SourceType] | None = None,
        method: Literal["similarity", "mmr"] = "similarity",
        fetch_k: int = 20,           # mmr candidate pool size
        lambda_mult: float = 0.5,    # mmr diversity weight (0=max diversity)
    ) -> list[MultimodalResult]:
        """
        Retrieve the top-k most relevant artefacts for `query`.

        Parameters
        ----------
        query:        Natural-language business question.
        k:            Number of results to return.
        source_types: If set, filter results to only these source types.
                      The retriever fetches extra candidates to compensate.
        method:       "similarity" (default) or "mmr".
        fetch_k:      Candidate pool for MMR pre-selection (ignored for similarity).
        lambda_mult:  MMR diversity weight (ignored for similarity).

        Returns
        -------
        list[MultimodalResult], length ≤ k.
        """
        # If filtering, oversample to have enough candidates after filtering
        retrieve_k = k * self._FILTER_OVERSAMPLE if source_types else k

        if method == "similarity":
            docs_and_scores = self._vs.similarity_search_with_score(
                query, k=retrieve_k
            )
        elif method == "mmr":
            # MMR returns Documents without scores; synthesise score=1.0 placeholder
            docs = self._vs.max_marginal_relevance_search(
                query, k=retrieve_k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
            docs_and_scores = [(d, 1.0) for d in docs]
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'similarity' or 'mmr'.")

        results = [
            self._to_result(doc, score)
            for doc, score in docs_and_scores
        ]

        if source_types:
            results = [r for r in results if r.source_type in source_types]

        top_k = results[:k]
        logger.debug(
            "retrieve(query=%r, k=%d, method=%s) → %d results",
            query[:60], k, method, len(top_k),
        )
        return top_k

    def retrieve_images_only(
        self, query: str, k: int = 4
    ) -> list[MultimodalResult]:
        """Convenience: only charts and table images."""
        return self.retrieve(query, k=k, source_types=["chart", "table", "pdf_image"])

    def retrieve_text_only(
        self, query: str, k: int = 4
    ) -> list[MultimodalResult]:
        """Convenience: only PDF text pages."""
        return self.retrieve(query, k=k, source_types=["pdf_text"])

    # ── LangChain retriever interface ─────────────────────────────────────

    def as_langchain_retriever(self, k: int = 4):
        """
        Return a LangChain BaseRetriever-compatible object.

        Allows this retriever to be dropped into any LCEL chain that
        expects a standard retriever interface, e.g.:

            chain = retriever.as_langchain_retriever() | prompt | llm

        The returned documents have the full MultimodalResult data packed
        into their metadata under the key "multimodal_result".
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        outer = self

        class _WrappedRetriever(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                results = outer.retrieve(query, k=k)
                return [
                    Document(
                        page_content=r.content,
                        metadata={**r.metadata, "score": r.score},
                    )
                    for r in results
                ]

        return _WrappedRetriever()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_result(doc: Document, score: float) -> MultimodalResult:
        meta = doc.metadata
        # FAISS returns L2 distance; convert to a similarity-like score
        # (lower distance = higher similarity).  We normalise to 0–1 range
        # by clamping: similarity = max(0, 1 - score).
        similarity = max(0.0, 1.0 - float(score))
        return MultimodalResult(
            content=doc.page_content,
            source_type=meta.get("source_type", "unknown"),
            file_path=meta.get("file_path", ""),
            file_name=meta.get("file_name", ""),
            title=meta.get("title", ""),
            score=similarity,
            metadata=meta,
        )
