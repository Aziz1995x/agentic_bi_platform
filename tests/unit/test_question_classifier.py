"""Unit tests for the question classifier.

These hit the real LLM (no mocking yet -- that's a deliberate choice for
Phase 3; mocking LLM calls is worth doing once you've seen what an
unmocked, flaky, real call looks like first). Mark them so they can be
skipped in fast local test runs later.
"""

from agentic_bi.classification.question_classifier import classify_question_bundle
import pytest

from agentic_bi.classification.question_classifier import classify_question
from agentic_bi.classification.schemas import InvestigationRoute

pytestmark = pytest.mark.llm  # register this marker in pyproject/pytest.ini


@pytest.mark.parametrize(
    "question,expected_route",
    [
        (
            "What is the definition of an active customer according to our business rules?",
            InvestigationRoute.RAG_ONLY,
        ),
        (
            "Which region has the highest return rate?",
            InvestigationRoute.SQL_ONLY,
        ),
        (
            "Are sales significantly different between regions?",
            InvestigationRoute.SQL_AND_PYTHON,
        ),
        (
            "Why did revenue decrease last quarter?",
            InvestigationRoute.FULL_INVESTIGATION,
        ),
    ],
)
def test_classifier_routes(question, expected_route):
    result = classify_question(question)
    assert result.route == expected_route
    assert result.reasoning  # non-empty


def test_classification_bundle_runs_concurrently():
    result = classify_question_bundle(
        "Why did revenue decrease last quarter?"
    )
    assert result.classification.route == InvestigationRoute.FULL_INVESTIGATION
    assert result.restated_question  # non-empty, LLM-generated restatement
