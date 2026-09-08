"""Tests for document loading and chunking.

No LLM calls here -- no pytest.mark.llm needed. This is pure I/O and
string-splitting logic, which is exactly why it should be tested with
plain, fast, deterministic assertions rather than skipped as "just
plumbing."
"""

from agentic_bi.rag.chunking import split_documents
from agentic_bi.rag.loaders import load_knowledge_base_documents


def test_loads_all_knowledge_base_documents():
    docs = load_knowledge_base_documents()
    assert len(docs) >= 4  # your 4 policy docs at minimum
    sources = {doc.metadata["source"] for doc in docs}
    assert "kpi_definitions.md" in sources
    assert "refund_and_returns_policy.md" in sources


def test_chunking_produces_multiple_chunks_per_document():
    docs = load_knowledge_base_documents()
    chunks = split_documents(docs)

    # Chunking should increase the total document count (each source
    # document is several hundred words, well over CHUNK_SIZE=800 chars).
    assert len(chunks) > len(docs)

    # Every chunk must retain its source metadata -- lose this and
    # citations break silently downstream.
    for chunk in chunks:
        print(chunk)
        assert "source" in chunk.metadata


def test_chunk_sizes_respect_configured_limit():
    docs = load_knowledge_base_documents()
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)

    # RecursiveCharacterTextSplitter can occasionally exceed chunk_size
    # slightly (it won't break a single "unsplittable" unit like one long
    # word), but should never wildly exceed it.
    for chunk in chunks:
        assert len(chunk.page_content) <= 900
