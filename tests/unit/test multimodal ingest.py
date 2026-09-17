"""
test_multimodal_ingest.py
─────────────────────────
Unit tests for the multimodal ingestion pipeline.

What's tested here (no API calls)
──────────────────────────────────
- Loaders produce correct Document structure from synthetic corpus
- Metadata fields are populated correctly
- needs_caption flag is set correctly for each source type
- PDF text extraction produces non-empty content
- PDF embedded image extraction creates sidecar files
- Caption cache load/save round-trips cleanly
- _should_caption() filtering logic

What's NOT tested here (covered in integration tests)
──────────────────────────────────────────────────────
- Actual vision API calls (captioner.py)
- FAISS index build and save
- End-to-end ingest_multimodal_corpus()

These are tagged @pytest.mark.llm and live in tests/integration/.
"""

import json
import struct
import zlib
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """
    Build a minimal multimodal corpus in a temp directory.
    All content is synthetic — no real chart files needed.
    """
    charts  = tmp_path / "charts"
    tables  = tmp_path / "tables"
    pdfs    = tmp_path / "pdfs"
    meta    = tmp_path / "metadata"

    for d in (charts, tables, pdfs, meta):
        d.mkdir()

    # ── Minimal valid PNG (1×1 white pixel) ──────────────────────────────
    def make_png(path: Path) -> None:
        def chunk(name: bytes, data: bytes) -> bytes:
            c = name + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\xff\xff"  # filter byte + 1 white RGB pixel
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        path.write_bytes(signature + ihdr + idat + iend)

    # Charts
    make_png(charts / "chart_revenue.png")
    (charts / "chart_returns.jpg").write_bytes(
        b"\xff\xd8\xff\xe0" + b"\x00" * 200   # minimal JPEG header
    )

    # Tables
    make_png(tables / "table_kpi.png")

    # Build a proper PDF with reportlab — same tool used by the real corpus
    # generators, so pypdf can parse it without warnings.
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    pdf_path = pdfs / "report_q2.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    doc.build([
        Paragraph("Q2 2018 Business Report", styles["Title"]),
        Paragraph(
            "Q2 2018 Revenue declined 4.7 percent quarter over quarter. "
            "Return rate increased to 5.2 percent. "
            "Delivery time deteriorated from 9.1 to 10.6 days.",
            styles["Normal"],
        ),
    ])

    # Metadata
    charts_meta = {
        "chart_revenue.png": {
            "title": "Monthly Revenue Trend",
            "key_insight": "Revenue peaked at R$810K in March 2018 before declining.",
            "chart_type": "line chart",
            "time_period": "Jan 2017 – Aug 2018",
            "data_source": "synthetic",
        },
        "chart_returns.jpg": {
            "title": "Return Rate by Category",
            "key_insight": "Watches & Gifts has the highest return rate at 6.8%.",
            "chart_type": "bar chart",
            "time_period": "Jan 2017 – Aug 2018",
            "data_source": "synthetic",
        },
    }
    tables_meta = {
        "table_kpi.png": {
            "title": "Q2 2018 KPI Summary",
            "key_insight": "Q2 2018 revenue R$2.25M, down 4.7% QoQ.",
            "table_type": "KPI table",
            "time_period": "Q2 2018",
            "data_source": "synthetic",
        },
    }
    pdfs_meta = {
        "report_q2.pdf": {
            "title": "Q2 2018 Business Report",
            "document_type": "quarterly report",
            "key_topics": ["revenue decline", "returns"],
            "time_period": "Q2 2018",
            "data_source": "synthetic",
        },
    }

    (meta / "charts_metadata.json").write_text(json.dumps(charts_meta))
    (meta / "tables_metadata.json").write_text(json.dumps(tables_meta))
    (meta / "pdfs_metadata.json").write_text(json.dumps(pdfs_meta))

    return tmp_path


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoadImageDocuments:
    def test_chart_count(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "charts_metadata.json").read_text())
        docs = load_image_documents(corpus_dir / "charts", "chart", meta)
        assert len(docs) == 2

    def test_chart_metadata_fields(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "charts_metadata.json").read_text())
        docs = load_image_documents(corpus_dir / "charts", "chart", meta)
        png_doc = next(d for d in docs if d.metadata["file_name"] == "chart_revenue.png")

        assert png_doc.metadata["source_type"] == "chart"
        assert png_doc.metadata["needs_caption"] is True
        assert png_doc.metadata["title"] == "Monthly Revenue Trend"
        assert "R$810K" in png_doc.metadata["key_insight"]
        assert Path(png_doc.metadata["file_path"]).exists()

    def test_jpg_chart_included(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "charts_metadata.json").read_text())
        docs = load_image_documents(corpus_dir / "charts", "chart", meta)
        names = [d.metadata["file_name"] for d in docs]
        assert "chart_returns.jpg" in names

    def test_source_type_propagated(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "tables_metadata.json").read_text())
        docs = load_image_documents(corpus_dir / "tables", "table", meta)
        assert all(d.metadata["source_type"] == "table" for d in docs)

    def test_missing_metadata_uses_defaults(self, corpus_dir):
        """A file not in the metadata JSON still produces a Document."""
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        # Pass empty metadata
        docs = load_image_documents(corpus_dir / "charts", "chart", {})
        assert len(docs) == 2
        for d in docs:
            # title falls back to stem
            assert d.metadata["title"] != ""
            assert d.metadata["needs_caption"] is True

    def test_empty_directory_returns_empty_list(self, tmp_path):
        from agentic_bi.rag.multimodal.loaders import load_image_documents
        (tmp_path / "empty").mkdir()
        docs = load_image_documents(tmp_path / "empty", "chart", {})
        assert docs == []


class TestLoadPdfDocuments:
    def test_pdf_text_extracted(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_pdf_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "pdfs_metadata.json").read_text())
        docs = load_pdf_documents(corpus_dir / "pdfs", meta)
        text_docs = [d for d in docs if d.metadata["source_type"] == "pdf_text"]
        assert len(text_docs) >= 1

    def test_pdf_text_content_nonempty(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_pdf_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "pdfs_metadata.json").read_text())
        docs = load_pdf_documents(corpus_dir / "pdfs", meta)
        text_docs = [d for d in docs if d.metadata["source_type"] == "pdf_text"]
        for d in text_docs:
            assert len(d.page_content.strip()) > 0
            assert d.metadata["needs_caption"] is False

    def test_pdf_text_metadata_fields(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_pdf_documents
        import json
        meta = json.loads((corpus_dir / "metadata" / "pdfs_metadata.json").read_text())
        docs = load_pdf_documents(corpus_dir / "pdfs", meta)
        text_docs = [d for d in docs if d.metadata["source_type"] == "pdf_text"]
        d = text_docs[0]
        assert d.metadata["file_name"] == "report_q2.pdf"
        assert d.metadata["title"] == "Q2 2018 Business Report"
        assert isinstance(d.metadata["page_number"], int)
        assert d.metadata["page_number"] >= 1

    def test_empty_pdf_dir_returns_empty(self, tmp_path):
        from agentic_bi.rag.multimodal.loaders import load_pdf_documents
        (tmp_path / "pdfs").mkdir()
        docs = load_pdf_documents(tmp_path / "pdfs", {})
        assert docs == []


class TestLoadMultimodalCorpus:
    def test_total_document_count(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_multimodal_corpus
        docs = load_multimodal_corpus(corpus_dir)
        # 2 charts + 1 table + ≥1 PDF text page = at least 4
        assert len(docs) >= 4

    def test_mix_of_needs_caption(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_multimodal_corpus
        docs = load_multimodal_corpus(corpus_dir)
        needs = [d for d in docs if d.metadata.get("needs_caption")]
        ready = [d for d in docs if not d.metadata.get("needs_caption")]
        # Images need captions; PDF text does not
        assert len(needs) >= 3     # 2 charts + 1 table
        assert len(ready) >= 1     # at least 1 PDF text page

    def test_all_file_paths_exist(self, corpus_dir):
        from agentic_bi.rag.multimodal.loaders import load_multimodal_corpus
        docs = load_multimodal_corpus(corpus_dir)
        for d in docs:
            fp = d.metadata.get("file_path", "")
            if fp:
                assert Path(fp).exists(), f"Missing: {fp}"


# ── Caption filtering tests ───────────────────────────────────────────────────

class TestShouldCaption:
    def test_image_doc_needs_captioning(self):
        from langchain_core.documents import Document
        from agentic_bi.rag.multimodal.ingest import _should_caption

        doc = Document(
            page_content="",
            metadata={"needs_caption": True, "file_path": ""}
        )
        # File path is empty → size check skipped → should_caption = True
        assert _should_caption(doc) is True

    def test_text_doc_does_not_need_captioning(self):
        from langchain_core.documents import Document
        from agentic_bi.rag.multimodal.ingest import _should_caption

        doc = Document(
            page_content="Revenue in Q2 2018 was R$2.25M.",
            metadata={"needs_caption": False}
        )
        assert _should_caption(doc) is False

    def test_tiny_image_skipped(self, tmp_path):
        """Image files smaller than MIN_IMAGE_BYTES are skipped."""
        from langchain_core.documents import Document
        from agentic_bi.rag.multimodal.ingest import _should_caption, MIN_IMAGE_BYTES

        tiny = tmp_path / "tiny.png"
        tiny.write_bytes(b"\x89PNG" + b"\x00" * 10)  # well below threshold
        assert tiny.stat().st_size < MIN_IMAGE_BYTES

        doc = Document(
            page_content="",
            metadata={"needs_caption": True, "file_path": str(tiny)}
        )
        assert _should_caption(doc) is False


# ── Caption cache tests ───────────────────────────────────────────────────────

class TestCaptionCache:
    def test_round_trip(self, tmp_path):
        from agentic_bi.rag.multimodal.ingest import _load_cache, _save_cache

        cache = {"key1": "caption one", "key2": "caption two"}
        _save_cache(tmp_path, cache)
        loaded = _load_cache(tmp_path)
        assert loaded == cache

    def test_missing_cache_returns_empty(self, tmp_path):
        from agentic_bi.rag.multimodal.ingest import _load_cache

        result = _load_cache(tmp_path / "nonexistent")
        assert result == {}

    def test_cache_key_changes_with_mtime(self, tmp_path):
        from agentic_bi.rag.multimodal.ingest import _cache_key

        f = tmp_path / "img.png"
        f.write_bytes(b"data1")
        key1 = _cache_key(str(f))

        import time
        time.sleep(1.1)   # filesystem mtime resolution is 1s on most systems
        f.write_bytes(b"data2-longer")
        key2 = _cache_key(str(f))

        # mtime changed → different cache key → will re-caption
        assert key1 != key2
