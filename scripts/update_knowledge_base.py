"""
scripts/update_knowledge_base.py
─────────────────────────────────
Appends new documents to the appropriate FAISS index.

Routing rules
─────────────
  .md / text-only .pdf  →  update_vectorstore()
        Text chunker (RecursiveCharacterTextSplitter). No image handling.
        Correct for policy docs, definitions, plain business text.

  Multimodal .pdf       →  update_multimodal_corpus()
        Full multimodal pipeline: extracts text pages AND embedded images,
        deduplicates images via pHash against standalone charts/tables,
        captions images via GPT-4o-mini vision, embeds everything with
        text-embedding-3-small, appends to the multimodal FAISS index.
        Correct for business reports, presentations, any PDF that may
        contain charts, tables, or diagrams alongside prose.

Why the distinction matters
────────────────────────────
PyPDFLoader + split_documents() is a text-only pipeline. On a multimodal
PDF it silently drops every embedded image. The content is lost with no
error or warning — the index receives only the prose pages, and any
chart or figure in the document becomes permanently unretrievable.

update_multimodal_corpus() handles both: text pages are indexed as-is,
embedded images are extracted, pHash-deduplicated against the standalone
corpus, captioned, and indexed alongside the text.

Run from project root:
    python scripts/update_knowledge_base.py
"""

from pathlib import Path

from langchain_core.documents import Document

from agentic_bi.rag.chunking import split_documents
from agentic_bi.rag.multimodal.ingest import update_multimodal_corpus
from agentic_bi.rag.vectorstore import update_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Text index: new policy document (.md) ─────────────────────────────────────
# seller_performance_policy.md is pure text — correct destination is the
# text FAISS index via the standard chunker.
md_path = PROJECT_ROOT / "documents" / "seller_performance_policy.md"
md_docs = [Document(
    page_content=md_path.read_text(encoding="utf-8"),
    metadata={"source": md_path.name},
)]
md_chunks = split_documents(md_docs)
update_vectorstore(md_chunks)
print(f"Text index updated: {len(md_chunks)} chunks from {md_path.name}")

# ── Multimodal index: new PDF business report ──────────────────────────────────
# customer_retention_framework.pdf is a mixed-content business report.
# It must go through the multimodal pipeline so that:
#   - Text pages are embedded normally
#   - Embedded images are extracted, pHash-deduplicated, captioned, and indexed
#   - Nothing is silently dropped
pdf_path = (
    PROJECT_ROOT / "documents" / "multimodal" / "pdfs"
    / "customer_retention_framework.pdf"
)
update_multimodal_corpus(
    new_pdf_paths=[pdf_path],
    metadata_lookup={
        "customer_retention_framework.pdf": {
            "title":         "Customer Retention Framework",
            "document_type": "business framework",
            "key_topics": [
                "customer segmentation",
                "repeat purchase rate",
                "lapsed customer win-back",
                "loyal customer retention",
                "retention KPIs",
                "active customer definition",
            ],
            "time_period": "March 2018",
            "data_source": "synthetic",
        }
    },
)
print(f"Multimodal index updated from {pdf_path.name}")
