"""
ingest.py
─────────
Orchestrates the complete multimodal ingestion pipeline:

    load corpus
        │
        ▼
    caption images          ← GPT-4o-mini vision, one API call per image
        │
        ▼
    build Documents         ← page_content = caption text (+ PDF text)
        │
        ▼
    embed with              ← text-embedding-3-small (same as text RAG)
    text-embedding-3-small
        │
        ▼
    save FAISS index        ← data/vectorstore/faiss_multimodal/

Key design decisions
────────────────────
1.  Separate FAISS index.
    The multimodal corpus is stored in faiss_multimodal/, distinct from
    the text-only faiss_index/.  This lets both indices exist side-by-side
    so the retrieval layer can query either or both, and means running
    multimodal ingestion does not invalidate text retrieval.

2.  Caption cache.
    API calls are expensive. A JSON sidecar file (caption_cache.json)
    stores file_path → caption so re-running ingestion only calls the
    vision API for new or changed files.  Cache is keyed by (file_path,
    file_mtime) to detect changes.

3.  Skip images that are too small.
    Embedded PDF images smaller than a threshold (default 5 KB) are
    typically decorative elements (logos, dividers) that carry no
    analytical content.  They are skipped rather than wasting an API call.

4.  Parallel captioning.
    Images are captioned with a thread pool (default: 4 workers) since
    each caption is a separate HTTP call with significant latency.
    Parallelism is bounded to avoid hitting the OpenAI rate limit.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agentic_bi.config.settings import get_settings
from agentic_bi.rag.embeddings import get_embeddings
from agentic_bi.rag.multimodal.captioner import caption_image_safe
from agentic_bi.rag.multimodal.loaders import load_multimodal_corpus

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_IMAGE_BYTES   = 5_000       # skip embedded PDF images smaller than this
MAX_CAPTION_WORKERS = 4         # parallel vision API calls
CAPTION_RETRY_DELAY = 2.0       # seconds between retries on rate-limit errors


# ── Caption cache ─────────────────────────────────────────────────────────────

def _cache_path(index_dir: Path) -> Path:
    return index_dir / "caption_cache.json"


def _load_cache(index_dir: Path) -> dict:
    p = _cache_path(index_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_cache(index_dir: Path, cache: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(index_dir), "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(file_path: str) -> str:
    """Cache key = path + mtime so changed files are re-captioned."""
    try:
        mtime = os.path.getmtime(file_path)
        return f"{file_path}::{mtime:.0f}"
    except FileNotFoundError:
        return file_path


# ── Image filtering ───────────────────────────────────────────────────────────

def _should_caption(doc: Document) -> bool:
    """
    Return True if this Document should go through the vision captioner.

    Filters out:
      - Documents that already have text (needs_caption=False)
      - Image files too small to contain meaningful content
    """
    if not doc.metadata.get("needs_caption", False):
        return False

    file_path = doc.metadata.get("file_path", "")
    if file_path and Path(file_path).exists():
        size = Path(file_path).stat().st_size
        if size < MIN_IMAGE_BYTES:
            logger.debug(
                "Skipping tiny image (%d bytes): %s",
                size, doc.metadata.get("file_name"),
            )
            return False

    return True


# ── Captioning with cache ─────────────────────────────────────────────────────

def _caption_one(
    doc: Document,
    cache: dict,
    index_dir: Path,
) -> tuple[Document, bool]:
    """
    Caption a single Document, using the cache if available.

    Returns (updated_doc, was_cached).
    Mutates doc.page_content in place.
    """
    file_path = doc.metadata.get("file_path", "")
    key = _cache_key(file_path)

    if key in cache:
        doc.page_content = cache[key]
        return doc, True

    extra_context = doc.metadata.get("key_insight") or None
    caption = caption_image_safe(
        Path(file_path),
        extra_context=extra_context,
        fallback=doc.metadata.get("key_insight", ""),
    )
    doc.page_content = caption
    cache[key] = caption
    return doc, False


def _run_captioning(
    docs_needing_caption: list[Document],
    cache: dict,
    index_dir: Path,
    max_workers: int,
) -> list[Document]:
    """
    Caption all images in parallel with a thread pool.
    Saves the cache after every batch to survive partial failures.
    """
    if not docs_needing_caption:
        return []

    results: list[Document] = [None] * len(docs_needing_caption)  # type: ignore[list-item]
    cache_hits = 0
    api_calls  = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_caption_one, doc, cache, index_dir): idx
            for idx, doc in enumerate(docs_needing_caption)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                doc, was_cached = future.result()
                results[idx] = doc
                if was_cached:
                    cache_hits += 1
                else:
                    api_calls += 1
            except Exception as exc:
                logger.error(
                    "Captioning failed for doc %d (%s): %s",
                    idx,
                    docs_needing_caption[idx].metadata.get("file_name", "?"),
                    exc,
                )
                # Keep original page_content (key_insight fallback)
                results[idx] = docs_needing_caption[idx]

    _save_cache(index_dir, cache)
    logger.info(
        "Captioning complete: %d cache hits, %d new API calls",
        cache_hits, api_calls,
    )
    return results


# ── FAISS index operations ────────────────────────────────────────────────────

def _multimodal_index_dir() -> Path:
    settings = get_settings()
    # Sibling of faiss_index at data/vectorstore/faiss_multimodal/
    return Path(settings.faiss_index_dir).parent / "faiss_multimodal"


def _build_and_save(docs: list[Document], index_dir: Path) -> FAISS:
    """Embed all documents and write the FAISS index to disk."""
    if not docs:
        raise ValueError("No documents to index — corpus appears empty.")

    embeddings = get_embeddings()
    logger.info("Embedding %d documents with %s ...", len(docs), embeddings)
    vectorstore = FAISS.from_documents(docs, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info("Multimodal FAISS index saved to %s", index_dir)
    return vectorstore


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_multimodal_corpus(
    multimodal_dir: Path | None = None,
    max_caption_workers: int = MAX_CAPTION_WORKERS,
    force_recaption: bool = False,
) -> FAISS:
    """
    Run the full multimodal ingestion pipeline.

    Parameters
    ----------
    multimodal_dir:
        Root of the multimodal corpus (charts/, tables/, pdfs/, metadata/).
        Defaults to documents/multimodal/ relative to project root.
    max_caption_workers:
        Number of parallel vision API calls. Reduce if hitting rate limits.
    force_recaption:
        If True, ignore the caption cache and re-caption everything.
        Useful after changing the caption prompt or switching vision models.

    Returns
    -------
    FAISS
        The built (and saved) vectorstore, ready for similarity_search().
    """
    settings = get_settings()

    if multimodal_dir is None:
        multimodal_dir = Path(settings.documents_dir) / "multimodal"

    index_dir = _multimodal_index_dir()
    cache     = {} if force_recaption else _load_cache(index_dir)

    logger.info("=== Multimodal RAG Ingestion ===")
    logger.info("Corpus dir : %s", multimodal_dir)
    logger.info("Index dir  : %s", index_dir)

    # ── Step 1: Load all corpus documents ────────────────────────────────
    all_docs = load_multimodal_corpus(multimodal_dir)
    if not all_docs:
        raise FileNotFoundError(
            f"No documents found in {multimodal_dir}. "
            "Run the corpus generators first:\n"
            "  python scripts/generate_corpus/generate_charts.py\n"
            "  python scripts/generate_corpus/generate_tables.py\n"
            "  python scripts/generate_corpus/generate_pdfs.py"
        )

    # ── Step 2: Caption images ────────────────────────────────────────────
    to_caption   = [d for d in all_docs if _should_caption(d)]
    text_ready   = [d for d in all_docs if not _should_caption(d)]

    logger.info(
        "Documents: %d to caption, %d text-ready",
        len(to_caption), len(text_ready),
    )

    captioned = _run_captioning(to_caption, cache, index_dir, max_caption_workers)

    # ── Step 3: Filter out documents with no usable content ──────────────
    all_ready = text_ready + captioned
    embeddable = [
        d for d in all_ready
        if d.page_content and len(d.page_content.strip()) >= 20
    ]
    skipped = len(all_ready) - len(embeddable)
    if skipped:
        logger.warning("Skipped %d documents with empty/short content", skipped)

    logger.info("Embedding %d documents ...", len(embeddable))

    # ── Step 4: Build and save FAISS index ───────────────────────────────
    vectorstore = _build_and_save(embeddable, index_dir)

    logger.info(
        "=== Ingestion complete: %d documents indexed ===",
        len(embeddable),
    )
    return vectorstore


def load_multimodal_vectorstore() -> FAISS:
    """
    Load a previously built multimodal FAISS index from disk.

    Raises FileNotFoundError if the index has not been built yet.
    Call ingest_multimodal_corpus() first.
    """
    index_dir = _multimodal_index_dir()
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Multimodal index not found at {index_dir}. "
            "Run: python scripts/ingest_multimodal.py"
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
