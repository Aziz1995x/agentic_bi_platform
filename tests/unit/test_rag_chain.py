import pytest

from agentic_bi.rag.chain import ask_knowledge_base

pytestmark = pytest.mark.llm


def test_rag_answers_single_document_question():
    result = ask_knowledge_base(
        "What is the restocking fee for a change of mind return?"
    )
    print(result)
    assert result.sufficient_context is True
    assert "10%" in result.answer or "10 %" in result.answer
    sources = {c.source for c in result.citations}
    assert "refund_and_returns_policy.md" in sources


def test_rag_answers_cross_document_question():
    result = ask_knowledge_base(
        "Can a customer be classified as enterprise even if they are not currently an active customer?"
    )
    print(result)
    assert result.sufficient_context is True
    sources = {c.source for c in result.citations}
    # This question genuinely requires both documents to answer correctly
    assert "customer_segmentation.md" in sources


def test_rag_flags_insufficient_context():
    result = ask_knowledge_base(
        "What is our policy on international shipping customs fees?"
    )
    print(result)
    # Not covered by any document -- should say so, not hallucinate
    assert result.sufficient_context is False
