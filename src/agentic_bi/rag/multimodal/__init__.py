"""
Multimodal RAG sub-package.

Public API
----------
    # Full corpus rebuild (first run or embedding model change)
    from agentic_bi.rag.multimodal import ingest_multimodal_corpus

    # Incremental update from raw file paths (recommended for new documents)
    from agentic_bi.rag.multimodal import update_multimodal_corpus

    # Incremental update from already-processed Documents
    from agentic_bi.rag.multimodal import update_multimodal_vectorstore

    # Load existing index for retrieval
    from agentic_bi.rag.multimodal import load_multimodal_vectorstore

Routing guide
─────────────
  New multimodal PDF or image file  →  update_multimodal_corpus()
  Already-captioned Documents       →  update_multimodal_vectorstore()
  First-time full build             →  ingest_multimodal_corpus()
  Query time                        →  load_multimodal_vectorstore()
"""

from agentic_bi.rag.multimodal.ingest import (
    ingest_multimodal_corpus,
    load_multimodal_vectorstore,
    update_multimodal_corpus,
    update_multimodal_vectorstore,
)

__all__ = [
    "ingest_multimodal_corpus",
    "load_multimodal_vectorstore",
    "update_multimodal_corpus",
    "update_multimodal_vectorstore",
]
