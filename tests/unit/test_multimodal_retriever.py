"""
test_multimodal_retriever.py
────────────────────────────
Unit tests for MultimodalRetriever and the eval metric functions.

No API calls. No FAISS index on disk.
The FAISS vectorstore is mocked with a lightweight stub that returns
pre-defined (Document, score) pairs.
"""

import math
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(file_name: str, source_type: str, content: str = "test content") -> Document:
    return Document(
        page_content=content,
        metadata={
            "file_name":   file_name,
            "file_path":   f"/corpus/{file_name}",
            "source_type": source_type,
            "title":       file_name.replace("_", " "),
            "needs_caption": False,
        },
    )


def _make_vs(docs_and_scores: list[tuple[Document, float]]) -> MagicMock:
    """Stub FAISS vectorstore that returns fixed results."""
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = docs_and_scores
    vs.max_marginal_relevance_search.return_value = [d for d, _ in docs_and_scores]
    return vs


# ── MultimodalResult tests ────────────────────────────────────────────────────

class TestMultimodalResult:
    def test_is_image_for_chart(self):
        from agentic_bi.rag.multimodal.retriever import MultimodalResult
        r = MultimodalResult(
            content="", source_type="chart", file_path="", file_name="x.png",
            title="", score=0.9,
        )
        assert r.is_image is True
        assert r.is_text is False

    def test_is_image_for_table(self):
        from agentic_bi.rag.multimodal.retriever import MultimodalResult
        r = MultimodalResult(
            content="", source_type="table", file_path="", file_name="x.png",
            title="", score=0.9,
        )
        assert r.is_image is True

    def test_is_image_for_pdf_image(self):
        from agentic_bi.rag.multimodal.retriever import MultimodalResult
        r = MultimodalResult(
            content="", source_type="pdf_image", file_path="", file_name="x.png",
            title="", score=0.8,
        )
        assert r.is_image is True

    def test_is_text_for_pdf_text(self):
        from agentic_bi.rag.multimodal.retriever import MultimodalResult
        r = MultimodalResult(
            content="revenue declined", source_type="pdf_text",
            file_path="", file_name="report.pdf", title="", score=0.85,
        )
        assert r.is_text is True
        assert r.is_image is False

    def test_score_normalisation(self):
        """FAISS L2 distance of 0.3 → similarity ≈ 0.7."""
        from agentic_bi.rag.multimodal.retriever import MultimodalRetriever
        doc = _make_doc("chart.png", "chart")
        vs  = _make_vs([(doc, 0.3)])
        retriever = MultimodalRetriever(vs)
        results = retriever.retrieve("revenue trend", k=1)
        assert abs(results[0].score - 0.7) < 1e-6

    def test_score_clamped_at_zero_for_large_distance(self):
        """FAISS L2 distance > 1.0 → similarity clamped to 0.0."""
        from agentic_bi.rag.multimodal.retriever import MultimodalRetriever
        doc = _make_doc("chart.png", "chart")
        vs  = _make_vs([(doc, 1.5)])
        retriever = MultimodalRetriever(vs)
        results = retriever.retrieve("revenue trend", k=1)
        assert results[0].score == 0.0


# ── MultimodalRetriever tests ─────────────────────────────────────────────────

class TestMultimodalRetriever:
    def _make_retriever(self, docs_and_scores):
        from agentic_bi.rag.multimodal.retriever import MultimodalRetriever
        return MultimodalRetriever(_make_vs(docs_and_scores))

    def test_returns_k_results(self):
        docs = [(d, 0.2) for d in [
            _make_doc("a.png", "chart"),
            _make_doc("b.png", "table"),
            _make_doc("c.png", "chart"),
            _make_doc("d.pdf", "pdf_text"),
        ]]
        retriever = self._make_retriever(docs)
        results = retriever.retrieve("query", k=4)
        assert len(results) == 4

    def test_result_fields_populated(self):
        doc = _make_doc("01_revenue.png", "chart", "Revenue peaked at R$810K")
        retriever = self._make_retriever([(doc, 0.15)])
        results = retriever.retrieve("revenue trend", k=1)
        r = results[0]
        assert r.file_name   == "01_revenue.png"
        assert r.source_type == "chart"
        assert r.content     == "Revenue peaked at R$810K"
        assert r.is_image    is True

    def test_source_type_filter_charts_only(self):
        docs = [
            (_make_doc("chart.png", "chart"),    0.1),
            (_make_doc("table.png", "table"),    0.2),
            (_make_doc("report.pdf", "pdf_text"), 0.3),
        ]
        retriever = self._make_retriever(docs * 3)  # oversample for filter
        results = retriever.retrieve("query", k=2, source_types=["chart"])
        assert all(r.source_type == "chart" for r in results)

    def test_retrieve_images_only_helper(self):
        docs = [
            (_make_doc("chart.png",  "chart"),    0.1),
            (_make_doc("table.png",  "table"),    0.2),
            (_make_doc("report.pdf", "pdf_text"), 0.3),
        ]
        retriever = self._make_retriever(docs * 3)
        results = retriever.retrieve_images_only("query", k=2)
        assert all(r.is_image for r in results)

    def test_retrieve_text_only_helper(self):
        docs = [
            (_make_doc("report.pdf", "pdf_text"), 0.1),
            (_make_doc("chart.png",  "chart"),    0.2),
        ]
        retriever = self._make_retriever(docs * 3)
        results = retriever.retrieve_text_only("query", k=1)
        assert all(r.is_text for r in results)

    def test_mmr_method_called(self):
        from agentic_bi.rag.multimodal.retriever import MultimodalRetriever
        vs = _make_vs([(_make_doc("x.png", "chart"), 0.1)])
        retriever = MultimodalRetriever(vs)
        retriever.retrieve("query", k=1, method="mmr")
        vs.max_marginal_relevance_search.assert_called_once()

    def test_invalid_method_raises(self):
        retriever = self._make_retriever([])
        with pytest.raises(ValueError, match="Unknown method"):
            retriever.retrieve("query", k=1, method="invalid")

    def test_as_langchain_retriever_returns_documents(self):
        doc = _make_doc("chart.png", "chart", "caption text")
        retriever = self._make_retriever([(doc, 0.2)])
        lc_retriever = retriever.as_langchain_retriever(k=1)
        results = lc_retriever.invoke("revenue")
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "caption text"
        assert "score" in results[0].metadata


# ── Eval metric tests ─────────────────────────────────────────────────────────

class TestEvalMetrics:
    """Test compute_metrics() in isolation — no retriever, no FAISS."""

    def _m(self, retrieved, expected, k):
        from tests.evaluation.run_multimodal_eval import compute_metrics
        return compute_metrics(retrieved, expected, k)

    # ── Hit Rate ─────────────────────────────────────────────────────────

    def test_hit_rate_perfect(self):
        hr, *_ = self._m(["a.png", "b.pdf"], ["a.png"], k=2)
        assert hr == 1.0

    def test_hit_rate_miss(self):
        hr, *_ = self._m(["c.png", "d.png"], ["a.png"], k=2)
        assert hr == 0.0

    # ── Recall ───────────────────────────────────────────────────────────

    def test_recall_full(self):
        _, recall, *_ = self._m(["a.png", "b.pdf"], ["a.png", "b.pdf"], k=2)
        assert recall == 1.0

    def test_recall_partial(self):
        _, recall, *_ = self._m(["a.png", "c.png"], ["a.png", "b.pdf"], k=2)
        assert abs(recall - 0.5) < 1e-9

    def test_recall_none(self):
        _, recall, *_ = self._m(["c.png", "d.png"], ["a.png", "b.pdf"], k=2)
        assert recall == 0.0

    # ── Precision ────────────────────────────────────────────────────────

    def test_precision_all_relevant(self):
        _, _, precision, *_ = self._m(["a.png", "b.pdf"], ["a.png", "b.pdf"], k=2)
        assert precision == 1.0

    def test_precision_half_relevant(self):
        _, _, precision, *_ = self._m(["a.png", "c.png"], ["a.png", "b.pdf"], k=2)
        assert abs(precision - 0.5) < 1e-9

    # ── Full Coverage ─────────────────────────────────────────────────────

    def test_full_coverage_pass(self):
        _, _, _, fc, *_ = self._m(["a.png", "b.pdf"], ["a.png", "b.pdf"], k=2)
        assert fc == 1.0

    def test_full_coverage_fail_when_one_missing(self):
        _, _, _, fc, *_ = self._m(["a.png", "c.png"], ["a.png", "b.pdf"], k=2)
        assert fc == 0.0

    # ── MRR ──────────────────────────────────────────────────────────────

    def test_mrr_first_rank(self):
        *_, mrr, _ = self._m(["a.png", "b.pdf"], ["a.png"], k=2)
        assert mrr == 1.0

    def test_mrr_second_rank(self):
        *_, mrr, _ = self._m(["c.png", "a.png"], ["a.png"], k=2)
        assert abs(mrr - 0.5) < 1e-9

    def test_mrr_miss(self):
        *_, mrr, _ = self._m(["c.png", "d.png"], ["a.png"], k=2)
        assert mrr == 0.0

    def test_mrr_windowed_to_k(self):
        """MRR must not find a hit beyond position k."""
        from tests.evaluation.run_multimodal_eval import compute_metrics
        # relevant doc is at rank 3, but k=2 → should not count
        *_, mrr, _ = compute_metrics(["c.png", "d.png", "a.png"], ["a.png"], k=2)
        assert mrr == 0.0

    # ── NDCG ─────────────────────────────────────────────────────────────

    def test_ndcg_perfect(self):
        *_, ndcg = self._m(["a.png", "b.pdf"], ["a.png", "b.pdf"], k=2)
        assert abs(ndcg - 1.0) < 1e-9

    def test_ndcg_single_relevant_at_rank_1(self):
        *_, ndcg = self._m(["a.png", "b.pdf"], ["a.png"], k=2)
        assert abs(ndcg - 1.0) < 1e-9   # only 1 relevant, it's first

    def test_ndcg_single_relevant_at_rank_2(self):
        *_, ndcg = self._m(["b.pdf", "a.png"], ["a.png"], k=2)
        # DCG = 1/log2(3) ≈ 0.631, IDCG = 1/log2(2) = 1.0 → NDCG ≈ 0.631
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert abs(ndcg - expected) < 1e-9

    def test_ndcg_zero_on_full_miss(self):
        *_, ndcg = self._m(["c.png", "d.png"], ["a.png"], k=2)
        assert ndcg == 0.0


# ── Eval cases sanity checks ──────────────────────────────────────────────────

class TestEvalCases:
    def test_ten_cases_defined(self):
        from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES
        assert len(MULTIMODAL_CASES) == 10

    def test_all_cases_have_unique_ids(self):
        from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES
        ids = [c.id for c in MULTIMODAL_CASES]
        assert len(ids) == len(set(ids))

    def test_all_expected_sources_nonempty(self):
        from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES
        for case in MULTIMODAL_CASES:
            assert len(case.expected_sources) >= 1, f"{case.id} has no expected sources"

    def test_multi_source_case_mm10_has_two_sources(self):
        from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES
        mm10 = next(c for c in MULTIMODAL_CASES if c.id == "MM10")
        assert len(mm10.expected_sources) == 2

    def test_all_queries_nonempty(self):
        from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES
        for case in MULTIMODAL_CASES:
            assert len(case.query.strip()) > 10, f"{case.id} query too short"
