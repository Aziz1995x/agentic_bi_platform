"""
scripts/build_text_index.py
────────────────────────────
Builds the text FAISS index from scratch by loading all knowledge base
documents, chunking them, embedding with text-embedding-3-small, and
saving both index.faiss and index.pkl to data/vectorstore/faiss_index/.

Run this:
  - On first setup (index does not exist yet)
  - After adding new .md or text documents to documents/
  - After changing the embedding model
  - After changing chunk_size or chunk_overlap
  - When index.pkl is missing (incomplete previous build)

For appending a small number of new documents to an existing index
without rebuilding from scratch, use scripts/update_knowledge_base.py
instead.

Usage
─────
    python scripts/build_text_index.py

    # See what would be indexed without making API calls
    python scripts/build_text_index.py --dry-run

    # Custom chunk settings
    python scripts/build_text_index.py --chunk-size 600 --chunk-overlap 80
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root: python scripts/build_text_index.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the text FAISS index from the knowledge base documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum characters per chunk (default: 800)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap characters between adjacent chunks (default: 100)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without making any API calls",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from agentic_bi.config.settings import get_settings
    from agentic_bi.rag.chunking import split_documents
    from agentic_bi.rag.loaders import load_knowledge_base_documents

    settings  = get_settings()
    index_dir = Path(settings.faiss_index_dir)

    # ── Load documents ────────────────────────────────────────────────
    logger.info("Loading knowledge base documents from %s ...", settings.documents_dir)
    docs = load_knowledge_base_documents()

    if not docs:
        logger.error(
            "No documents found in %s. "
            "Add .md or .txt files to the documents directory first.",
            settings.documents_dir,
        )
        sys.exit(1)

    logger.info("Loaded %d documents", len(docs))

    # ── Chunk ─────────────────────────────────────────────────────────
    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(
        "Chunked into %d chunks (size=%d, overlap=%d)",
        len(chunks), args.chunk_size, args.chunk_overlap,
    )

    # ── Dry run report ────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDry run — no API calls made")
        print(f"Documents dir : {settings.documents_dir}")
        print(f"Index dir     : {index_dir}")
        print(f"Documents     : {len(docs)}")
        print(f"Chunks        : {len(chunks)}")
        print(f"Chunk size    : {args.chunk_size}")
        print(f"Chunk overlap : {args.chunk_overlap}")
        print(f"\nDocuments that would be indexed:")
        sources = sorted({c.metadata.get('source', 'unknown') for c in chunks})
        for source in sources:
            count = sum(1 for c in chunks if c.metadata.get('source') == source)
            print(f"  {source:<50} {count:>3} chunks")
        print(f"\nEmbedding model : {settings.embedding_model}")
        print(f"Estimated cost  : ~{len(chunks) * 300} tokens "
              f"(~${len(chunks) * 300 * 0.00000002:.4f} at $0.02/1M tokens)")
        return

    # ── Build and save ────────────────────────────────────────────────
    logger.info("Building FAISS index — this makes embedding API calls ...")

    from agentic_bi.rag.vectorstore import build_and_save_knowledge_base

    vectorstore = build_and_save_knowledge_base(chunks)

    # ── Verify both files exist ───────────────────────────────────────
    faiss_file = index_dir / "index.faiss"
    pkl_file   = index_dir / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        logger.error(
            "Index files missing after build — something went wrong.\n"
            "  index.faiss : %s\n"
            "  index.pkl   : %s",
            "OK" if faiss_file.exists() else "MISSING",
            "OK" if pkl_file.exists()   else "MISSING",
        )
        sys.exit(1)

    # ── Smoke test ────────────────────────────────────────────────────
    print("\n── Smoke test ──────────────────────────────────────────────")
    results = vectorstore.similarity_search("active customer definition", k=3)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {source}")
        print(f"       {doc.page_content[:100].strip()} ...")
    print()

    logger.info(
        "=== Text index build complete ===\n"
        "  index.faiss : %s (%.1f KB)\n"
        "  index.pkl   : %s (%.1f KB)\n"
        "  Chunks      : %d\n"
        "  Documents   : %d",
        faiss_file,   faiss_file.stat().st_size / 1024,
        pkl_file,     pkl_file.stat().st_size / 1024,
        len(chunks),
        len(docs),
    )


if __name__ == "__main__":
    main()
