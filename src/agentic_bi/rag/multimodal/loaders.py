"""
loaders.py
──────────
Converts the three multimodal artefact types into lists of
langchain_core.documents.Document objects ready for captioning and
embedding.

Each Document carries:
  - page_content: the text that will be embedded (caption for images,
                  extracted text for PDF text pages)
  - metadata dict with:
      source_type:   "chart" | "table" | "pdf_text" | "pdf_image"
      file_path:     absolute path to the original file (str)
      file_name:     filename only, e.g. "01_monthly_revenue_trend.png"
      title:         human-readable title from metadata JSON (if available)
      key_insight:   pre-written insight from metadata JSON (if available)
      page_number:   page index (PDFs only)
      image_index:   position of image within PDF page (pdf_image only)
      needs_caption: True  → page_content is a placeholder; captioner
                             must replace it before embedding
                    False → page_content is already usable text

Two-pass design
───────────────
loaders.py produces Documents with needs_caption=True for every image.
ingest.py reads that flag and calls captioner.caption_image_safe() to
fill page_content before building the FAISS index.

This separation means:
  - loaders.py has no API calls and is fully testable without credentials
  - captioner.py has one responsibility: image → text
  - ingest.py orchestrates the pipeline end-to-end

Deduplication
─────────────
PDF reports often embed the same charts that exist as standalone files
in charts/ or tables/. Indexing both produces duplicate semantic content
that confuses the retriever — the standalone file and its PDF-extracted
copy compete for the same query, and neither reliably wins.

Option B deduplication: when loading PDF embedded images, compute a
perceptual hash (pHash) of the extracted bytes and compare against the
hashes of all known standalone images. If the distance is within
PHASH_DUPLICATE_THRESHOLD, the embedded image is a duplicate and is
skipped. Images with no standalone match are kept and indexed normally.

Why perceptual hash rather than MD5?
─────────────────────────────────────
PDF embedding re-encodes images at different quality settings. The raw
bytes differ (MD5 would miss the duplicate) but the visual content is
identical (pHash distance ≈ 0–3, well within threshold 10).
"""

import io
import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Perceptual hash distance threshold for duplicate detection.
# Identical images: distance 0. Re-encoded same image: typically < 5.
# Visually similar but distinct images: typically > 15.
PHASH_DUPLICATE_THRESHOLD = 10


# ── Metadata loading ──────────────────────────────────────────────────────────

def _load_metadata_json(meta_dir: Path, filename: str) -> dict:
    """Load a metadata JSON file; return empty dict if missing."""
    path = meta_dir / filename
    if not path.exists():
        logger.warning("Metadata file not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


# ── Perceptual hashing ────────────────────────────────────────────────────────

def _phash_bytes(image_bytes: bytes) -> "imagehash.ImageHash | None":
    """
    Compute the perceptual hash of raw image bytes.

    Returns None if the bytes cannot be decoded as an image (e.g. a tiny
    decorative element that PIL rejects).  Callers treat None as
    non-duplicate so the image is kept rather than silently dropped.
    """
    try:
        import imagehash
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return imagehash.phash(img)
    except Exception as exc:
        logger.debug("Could not phash image bytes (%d bytes): %s", len(image_bytes), exc)
        return None


def _phash_file(image_path: Path) -> "imagehash.ImageHash | None":
    """Compute the perceptual hash of an image file on disk."""
    try:
        import imagehash
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)
    except Exception as exc:
        logger.debug("Could not phash %s: %s", image_path.name, exc)
        return None


def _build_standalone_hash_set(image_dirs: list[Path]) -> list["imagehash.ImageHash"]:
    """
    Return a list of perceptual hashes for every image in the given dirs.

    Called once at corpus-load time; the resulting list is passed into
    load_pdf_documents() so each extracted PDF image can be checked
    against it in O(n_standalone) time.

    Returns a list (not a set) because ImageHash objects are not hashable
    in the Python set sense — comparison uses the subtraction operator
    for Hamming distance.
    """
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    hashes: list = []

    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in supported:
                continue
            h = _phash_file(img_path)
            if h is not None:
                hashes.append(h)

    logger.info(
        "Built standalone hash set: %d hashes from %d dirs",
        len(hashes), len(image_dirs),
    )
    return hashes


def _is_duplicate(
    image_bytes: bytes,
    standalone_hashes: list,
    threshold: int = PHASH_DUPLICATE_THRESHOLD,
) -> bool:
    """
    Return True if image_bytes is perceptually identical to any standalone image.

    Distance 0  = bit-for-bit identical visual content.
    Distance ≤10 = same image, possibly re-encoded.
    Distance >15 = visually distinct.
    """
    if not standalone_hashes:
        return False

    candidate = _phash_bytes(image_bytes)
    if candidate is None:
        return False  # can't hash → assume not duplicate, keep it

    for known_hash in standalone_hashes:
        if (candidate - known_hash) <= threshold:
            return True

    return False


# ── Chart / table image loaders ───────────────────────────────────────────────

def load_image_documents(
    image_dir: Path,
    source_type: str,
    metadata_lookup: dict,
) -> list[Document]:
    """
    Produce one Document per image file in `image_dir`.

    page_content is set to the key_insight from metadata if available
    (used as a caption placeholder and as extra_context for the captioner).
    The real caption is filled in by ingest.py.
    """
    docs = []
    supported = {".png", ".jpg", ".jpeg", ".webp"}

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in supported
    )

    if not image_files:
        logger.warning("No image files found in %s", image_dir)
        return docs

    for img_path in image_files:
        meta = metadata_lookup.get(img_path.name, {})
        doc = Document(
            page_content=meta.get("key_insight", ""),
            metadata={
                "source_type":   source_type,
                "file_path":     str(img_path.resolve()),
                "file_name":     img_path.name,
                "title":         meta.get("title", img_path.stem),
                "key_insight":   meta.get("key_insight", ""),
                "chart_type":    meta.get("chart_type", ""),
                "time_period":   meta.get("time_period", ""),
                "data_source":   meta.get("data_source", "synthetic"),
                "needs_caption": True,
            },
        )
        docs.append(doc)
        logger.debug("Loaded %s image: %s", source_type, img_path.name)

    logger.info("Loaded %d %s documents from %s", len(docs), source_type, image_dir)
    return docs


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf_documents(
    pdf_dir: Path,
    metadata_lookup: dict,
    min_text_length: int = 80,
    standalone_image_dirs: list[Path] | None = None,
) -> list[Document]:
    """
    Extract both text and embedded images from every PDF in `pdf_dir`.

    Returns a mixed list of Documents:
      - PDF text chunks  (needs_caption=False, page_content = extracted text)
      - PDF embedded images (needs_caption=True, page_content = placeholder)
        — only those NOT already present as standalone files

    Parameters
    ----------
    pdf_dir:
        Directory containing *.pdf files.
    metadata_lookup:
        filename → metadata dict from pdfs_metadata.json.
    min_text_length:
        Minimum character count for a text page to be indexed.
        Pages shorter than this are typically cover pages or headers.
    standalone_image_dirs:
        Directories containing standalone chart/table images (charts/,
        tables/).  When provided, each extracted PDF image is compared
        against these via perceptual hashing.  Duplicates are skipped.
        Pass None to disable deduplication (not recommended in production).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF loading. "
            "Install with: pip install pypdf"
        ) from e

    # Build the standalone hash set once before iterating PDFs
    standalone_hashes: list = []
    if standalone_image_dirs:
        standalone_hashes = _build_standalone_hash_set(standalone_image_dirs)

    docs = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return docs

    total_skipped_duplicates = 0

    for pdf_path in pdf_files:
        file_meta = metadata_lookup.get(pdf_path.name, {})
        logger.info("Loading PDF: %s", pdf_path.name)

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to read %s: %s", pdf_path.name, exc)
            continue

        for page_idx, page in enumerate(reader.pages):

            # ── Extract text ──────────────────────────────────────────────
            text = (page.extract_text() or "").strip()
            if len(text) >= min_text_length:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_type":   "pdf_text",
                        "file_path":     str(pdf_path.resolve()),
                        "file_name":     pdf_path.name,
                        "title":         file_meta.get("title", pdf_path.stem),
                        "page_number":   page_idx + 1,
                        "key_topics":    file_meta.get("key_topics", []),
                        "document_type": file_meta.get("document_type", "report"),
                        "time_period":   file_meta.get("time_period", ""),
                        "data_source":   file_meta.get("data_source", "synthetic"),
                        "needs_caption": False,
                    },
                ))
                logger.debug(
                    "  page %d: extracted %d chars of text",
                    page_idx + 1, len(text),
                )

            # ── Extract embedded images ───────────────────────────────────
            if not hasattr(page, "images"):
                continue

            for img_idx, img_obj in enumerate(page.images):
                img_name = (
                    f"{pdf_path.stem}_page{page_idx + 1}_img{img_idx}.png"
                )

                try:
                    img_bytes = img_obj.data
                except Exception as exc:
                    logger.warning("Could not read image %s: %s", img_name, exc)
                    continue

                # ── Deduplication check ───────────────────────────────────
                if standalone_hashes and _is_duplicate(img_bytes, standalone_hashes):
                    logger.debug(
                        "  page %d: skipping duplicate embedded image %s",
                        page_idx + 1, img_name,
                    )
                    total_skipped_duplicates += 1
                    continue

                # ── Save and index non-duplicate ──────────────────────────
                img_save_path = pdf_dir / "_extracted" / img_name
                img_save_path.parent.mkdir(exist_ok=True)

                try:
                    img_save_path.write_bytes(img_bytes)
                except Exception as exc:
                    logger.warning(
                        "  Could not save embedded image %s: %s",
                        img_name, exc,
                    )
                    continue

                docs.append(Document(
                    page_content="",
                    metadata={
                        "source_type":   "pdf_image",
                        "file_path":     str(img_save_path.resolve()),
                        "file_name":     img_name,
                        "parent_pdf":    pdf_path.name,
                        "title":         file_meta.get("title", pdf_path.stem),
                        "page_number":   page_idx + 1,
                        "image_index":   img_idx,
                        "document_type": file_meta.get("document_type", "report"),
                        "time_period":   file_meta.get("time_period", ""),
                        "data_source":   file_meta.get("data_source", "synthetic"),
                        "needs_caption": True,
                    },
                ))
                logger.debug(
                    "  page %d: indexed non-duplicate embedded image → %s",
                    page_idx + 1, img_name,
                )

    if total_skipped_duplicates:
        logger.info(
            "Deduplication: skipped %d embedded images already present "
            "as standalone files",
            total_skipped_duplicates,
        )

    logger.info(
        "PDF loading complete: %d total documents from %d PDFs",
        len(docs), len(pdf_files),
    )
    return docs


# ── Top-level corpus loader ───────────────────────────────────────────────────

def load_multimodal_corpus(multimodal_dir: Path) -> list[Document]:
    """
    Load the complete multimodal corpus from the standard directory layout:

        multimodal_dir/
            charts/      ← PNG and JPG chart images
            tables/      ← PNG table images
            pdfs/        ← mixed-content PDF reports
            metadata/
                charts_metadata.json
                tables_metadata.json
                pdfs_metadata.json

    Returns a flat list of all Documents (image placeholders + PDF text).
    Documents with needs_caption=True must be captioned before embedding.

    Deduplication is enabled by default: PDF embedded images that are
    perceptually identical to any standalone chart or table image are
    skipped. This prevents the same visual content appearing twice in the
    index with different captions, which would confuse retrieval ranking.
    """
    meta_dir   = multimodal_dir / "metadata"
    charts_dir = multimodal_dir / "charts"
    tables_dir = multimodal_dir / "tables"
    pdfs_dir   = multimodal_dir / "pdfs"

    charts_meta = _load_metadata_json(meta_dir, "charts_metadata.json")
    tables_meta = _load_metadata_json(meta_dir, "tables_metadata.json")
    pdfs_meta   = _load_metadata_json(meta_dir, "pdfs_metadata.json")

    all_docs: list[Document] = []

    if charts_dir.exists():
        all_docs.extend(load_image_documents(charts_dir, "chart", charts_meta))

    if tables_dir.exists():
        all_docs.extend(load_image_documents(tables_dir, "table", tables_meta))

    if pdfs_dir.exists():
        # Pass standalone dirs so PDF image extraction deduplicates
        # against the charts and tables we just loaded above.
        all_docs.extend(load_pdf_documents(
            pdfs_dir,
            pdfs_meta,
            standalone_image_dirs=[charts_dir, tables_dir],
        ))

    needs_caption = sum(1 for d in all_docs if d.metadata.get("needs_caption"))
    logger.info(
        "Corpus loaded: %d total documents (%d need captioning, %d text-ready)",
        len(all_docs),
        needs_caption,
        len(all_docs) - needs_caption,
    )
    return all_docs
