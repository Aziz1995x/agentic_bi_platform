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
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Metadata loading ──────────────────────────────────────────────────────────

def _load_metadata_json(meta_dir: Path, filename: str) -> dict:
    """Load a metadata JSON file; return empty dict if missing."""
    path = meta_dir / filename
    if not path.exists():
        logger.warning("Metadata file not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


# ── Chart / table image loaders ───────────────────────────────────────────────

def load_image_documents(
    image_dir: Path,
    source_type: str,          # "chart" or "table"
    metadata_lookup: dict,     # filename → metadata dict
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
            page_content=meta.get("key_insight", ""),   # placeholder
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
) -> list[Document]:
    """
    Extract both text and embedded images from every PDF in `pdf_dir`.

    Returns a mixed list of Documents:
      - PDF text chunks (needs_caption=False, page_content = extracted text)
      - PDF embedded images (needs_caption=True, page_content = placeholder)

    Why split text and images separately?
    ──────────────────────────────────────
    A PDF report contains two kinds of knowledge:
      1. The prose analysis (text) — already embeddable as-is
      2. The embedded charts/tables (images) — need captioning

    Embedding the raw text without also captioning the images means
    queries like "show me the return rate chart" can only match the
    text caption below the chart ("Figure 4: Return rate by category"),
    not the chart itself.  By extracting and captioning embedded images
    separately, both paths work.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF loading. "
            "Install with: pip install pypdf"
        ) from e

    docs = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return docs

    for pdf_path in pdf_files:
        file_meta = metadata_lookup.get(pdf_path.name, {})
        logger.info("Loading PDF: %s", pdf_path.name)

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to read %s: %s", pdf_path.name, exc)
            continue

        for page_idx, page in enumerate(reader.pages):
            # ── Extract text ─────────────────────────────────────────────
            text = page.extract_text() or ""
            text = text.strip()

            if len(text) >= min_text_length:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_type":  "pdf_text",
                        "file_path":    str(pdf_path.resolve()),
                        "file_name":    pdf_path.name,
                        "title":        file_meta.get("title", pdf_path.stem),
                        "page_number":  page_idx + 1,
                        "key_topics":   file_meta.get("key_topics", []),
                        "document_type": file_meta.get("document_type", "report"),
                        "time_period":  file_meta.get("time_period", ""),
                        "data_source":  file_meta.get("data_source", "synthetic"),
                        "needs_caption": False,
                    },
                ))
                logger.debug(
                    "  page %d: extracted %d chars of text",
                    page_idx + 1, len(text),
                )

            # ── Extract embedded images ───────────────────────────────────
            # pypdf exposes images via page.images (pypdf >= 3.x)
            if hasattr(page, "images"):
                for img_idx, img_obj in enumerate(page.images):
                    # Save extracted image to a temp path alongside the PDF
                    img_name = (
                        f"{pdf_path.stem}_page{page_idx+1}_img{img_idx}.png"
                    )
                    img_save_path = pdf_path.parent / "_extracted" / img_name
                    img_save_path.parent.mkdir(exist_ok=True)

                    try:
                        img_save_path.write_bytes(img_obj.data)
                    except Exception as exc:
                        logger.warning(
                            "  Could not save embedded image %s: %s",
                            img_name, exc,
                        )
                        continue

                    docs.append(Document(
                        page_content="",    # captioner will fill this
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
                        "  page %d: extracted embedded image → %s",
                        page_idx + 1, img_name,
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
            charts/          ← PNG and JPG chart images
            tables/          ← PNG table images
            pdfs/            ← mixed-content PDF reports
            metadata/
                charts_metadata.json
                tables_metadata.json
                pdfs_metadata.json

    Returns a flat list of all Documents (image placeholders + PDF text).
    Documents with needs_caption=True must be captioned before embedding.
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
        all_docs.extend(load_pdf_documents(pdfs_dir, pdfs_meta))

    needs_caption = sum(1 for d in all_docs if d.metadata.get("needs_caption"))
    logger.info(
        "Corpus loaded: %d total documents (%d need captioning, %d text-ready)",
        len(all_docs),
        needs_caption,
        len(all_docs) - needs_caption,
    )
    return all_docs
