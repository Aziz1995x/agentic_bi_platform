"""
ingest_multimodal.py
────────────────────
CLI entry point for multimodal RAG ingestion.

Usage
─────
    # Standard run (uses caption cache)
    python scripts/ingest_multimodal.py

    # Force re-caption everything (e.g. after changing the prompt)
    python scripts/ingest_multimodal.py --force-recaption

    # Custom corpus directory
    python scripts/ingest_multimodal.py --corpus-dir /path/to/multimodal

    # Reduce parallelism if hitting rate limits
    python scripts/ingest_multimodal.py --workers 2

    # Dry-run: show what would be ingested without calling the API
    python scripts/ingest_multimodal.py --dry-run

Expected output
───────────────
    === Multimodal RAG Ingestion ===
    Corpus dir : documents/multimodal
    Index dir  : data/vectorstore/faiss_multimodal
    Loaded 10 chart documents
    Loaded 5 table documents
    Loaded 31 PDF documents (text + embedded images)
    Documents: 23 to caption, 23 text-ready
    Captioning complete: 0 cache hits, 23 new API calls
    Embedding 46 documents ...
    Multimodal FAISS index saved to data/vectorstore/faiss_multimodal
    === Ingestion complete: 46 documents indexed ===

Cost estimate (first run, no cache)
────────────────────────────────────
    Vision calls : ~23 images × ~$0.001 each = ~$0.02
    Embedding    : ~46 chunks × ~500 tokens  = ~23K tokens ≈ $0.00005
    Total        : < $0.05 for the full corpus
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root: python scripts/ingest_multimodal.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest multimodal corpus into FAISS vectorstore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to multimodal corpus root (default: documents/multimodal)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel caption API calls (default: 4)",
    )
    parser.add_argument(
        "--force-recaption",
        action="store_true",
        help="Ignore cache and re-caption all images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load corpus and report what would be ingested, without API calls",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        _dry_run(args.corpus_dir)
        return

    from agentic_bi.rag.multimodal.ingest import ingest_multimodal_corpus

    vectorstore = ingest_multimodal_corpus(
        multimodal_dir=args.corpus_dir,
        max_caption_workers=args.workers,
        force_recaption=args.force_recaption,
    )

    # Quick smoke-test retrieval
    print("\n── Smoke test ──────────────────────────────────────────")
    results = vectorstore.similarity_search("return rate by product category", k=3)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("file_name", "unknown")
        stype  = doc.metadata.get("source_type", "?")
        print(f"  [{i}] ({stype}) {source}")
        print(f"       {doc.page_content[:120].strip()}...")
    print()


def _dry_run(corpus_dir: Path | None) -> None:
    """Report corpus contents without making any API calls."""
    from agentic_bi.config.settings import get_settings
    from agentic_bi.rag.multimodal.loaders import load_multimodal_corpus

    settings = get_settings()
    if corpus_dir is None:
        corpus_dir = Path(settings.documents_dir) / "multimodal"

    print(f"Corpus dir: {corpus_dir}")
    docs = load_multimodal_corpus(corpus_dir)

    by_type: dict[str, list] = {}
    for d in docs:
        t = d.metadata.get("source_type", "unknown")
        by_type.setdefault(t, []).append(d)

    print(f"\nTotal documents: {len(docs)}")
    for stype, ds in sorted(by_type.items()):
        needs = sum(1 for d in ds if d.metadata.get("needs_caption"))
        print(f"  {stype:<15} {len(ds):>3} documents  ({needs} need captioning)")

    print("\nFiles that would be captioned:")
    for d in docs:
        if d.metadata.get("needs_caption"):
            print(f"  {d.metadata.get('file_name', '?')}")


if __name__ == "__main__":
    main()
