"""ColBERT (late-interaction) retriever via RAGatouille.

Unlike every other retriever in this project, ColBERT does not reuse your
existing FAISS index or embeddings factory -- it builds and queries its
own token-level index entirely separately. This is worth sitting with as
an architectural fact: adopting ColBERT in a real system means running
and maintaining a second retrieval subsystem alongside your dense one,
not just swapping a parameter -- a real operational cost that a
lightweight technique like MMR or Hybrid does not carry.
"""

from pathlib import Path

from langchain_core.documents import Document
from ragatouille import RAGPretrainedModel


def build_colbert_index(chunks: list[Document], index_name: str = "policy_docs_colbert"):
    """Builds a ColBERT index from your existing chunks (same 800/100
    RecursiveCharacterTextSplitter output as FAISS -- chunk CONTENT is
    identical across techniques, only the indexing/scoring mechanism
    differs, which keeps this a fair comparison).

    This downloads colbert-ir/colbertv2.0 on first use (~440MB) and
    performs real per-token encoding across every chunk -- noticeably
    slower to build than embedding your chunks for FAISS. Budget for that
    the first time you run this.
    """
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    index_path = RAG.index(
        collection=texts,
        document_metadatas=metadatas,
        index_name=index_name,
        max_document_length=256,  # ColBERT truncates beyond this per-passage token limit
        split_documents=False,     # already chunked -- don't let RAGatouille re-split
    )
    return RAG, index_path


def colbert_search(RAG: RAGPretrainedModel, query: str, k: int = 4) -> list[Document]:
    """Runs a ColBERT MaxSim search and converts results back into
    LangChain Documents so they work with the same eval harness as
    every other retriever in this phase."""
    results = RAG.search(query=query, k=k)
    return [
        Document(
            page_content=r["content"],
            metadata=r.get("document_metadata", {}),
        )
        for r in results
    ]
