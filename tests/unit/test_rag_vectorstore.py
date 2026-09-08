"""Tests for FAISS build/save/load and similarity search.

These make real embedding API calls (not mocked) -- same philosophy as
Phase 3's LLM tests. Mark them so they can be excluded from a fast local
loop later.
"""

import pytest

from agentic_bi.rag.chunking import split_documents
from agentic_bi.rag.loaders import load_knowledge_base_documents
from agentic_bi.rag.vectorstore import build_and_save_knowledge_base, load_vectorstore

pytestmark = pytest.mark.llm  # embedding calls are external API calls too


@pytest.fixture(scope="module")
def knowledge_base():
    docs = load_knowledge_base_documents()
    chunks = split_documents(docs)
    return build_and_save_knowledge_base(chunks)


def test_similarity_search_returns_relevant_chunk(knowledge_base):
    results = knowledge_base.similarity_search(
        "What is the restocking fee for change of mind returns?", k=3
    )
    assert len(results) == 3
    # At least one of the top-3 results should come from the refund policy doc
    sources = {r.metadata["source"] for r in results}
    assert "refund_and_returns_policy.md" in sources


def test_saved_index_reloads_with_same_results(knowledge_base):
    reloaded = load_vectorstore()
    results = reloaded.similarity_search("definition of active customer", k=1)
    assert results[0].metadata["source"] == "kpi_definitions.md"
