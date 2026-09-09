"""Fixed retrieval-quality test set for the RAG knowledge base.

Locked in for Phase 5 -- every retrieval technique (MMR, MultiQuery,
hybrid, reranking, ParentDocumentRetriever) gets measured against this
exact set so improvements are attributable to a specific change, not
vibes. Do not edit expected_sources casually once comparisons start --
if a case turns out to be mis-specified, note it explicitly rather than
quietly changing the ground truth.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievalTestCase:
    id: int
    question: str
    expected_sources: set[str]  # empty set = no document should sufficiently answer this
    note: str = ""


RETRIEVAL_TEST_CASES: list[RetrievalTestCase] = [
    RetrievalTestCase(
        1,
        "What is the restocking fee percentage?",
        {"refund_and_returns_policy.md"},
        "Baseline sanity check -- exact phrase match, should be trivial.",
    ),
    RetrievalTestCase(
        2,
        "How long does a customer have to request a return?",
        {"refund_and_returns_policy.md"},
        "Query phrasing differs from doc phrasing ('7 calendar days').",
    ),
    RetrievalTestCase(
        3,
        "What counts as an active customer?",
        {"kpi_definitions.md"},
        "Vocabulary mismatch: 'what counts as' vs 'is considered active if'.",
    ),
    RetrievalTestCase(
        4,
        "Is a customer still enterprise if they haven't ordered in a while?",
        {"kpi_definitions.md", "customer_segmentation.md"},
        "Cross-document synthesis -- regression guard, already passes at k=4.",
    ),
    RetrievalTestCase(
        5,
        "Does freight count as revenue?",
        {"kpi_definitions.md", "pricing_and_freight_policy.md"},
        "Cross-document: Revenue definition vs freight's exclusion rationale.",
    ),
    RetrievalTestCase(
        6,
        "What's the minimum order volume needed before a return rate is meaningful?",
        {"kpi_definitions.md"},
        "Buried numeric fact (20 items) inside a longer Return Rate section.",
    ),
    RetrievalTestCase(
        7,
        "Are damaged items subject to the restocking fee?",
        {"refund_and_returns_policy.md"},
        "Trick question -- adjacent near-duplicate rules (fee applies only to change-of-mind).",
    ),
    RetrievalTestCase(
        8,
        "How is a seller's region determined?",
        {"kpi_definitions.md", "pricing_and_freight_policy.md"},
        "Same fact (sellers -> employees -> regions) stated in two docs for different purposes.",
    ),
    RetrievalTestCase(
        9,
        "What happens if a customer returns an item after 30 days?",
        {"refund_and_returns_policy.md"},
        "Borderline -- doc has no 30-day-specific rule, only 7-day window + exception process.",
    ),
    RetrievalTestCase(
        10,
        "What is our policy on customs fees for international orders?",
        set(),
        "Negative control -- no document covers this. Retrieval will still return "
        "something (retrievers always return k results); the pass condition here is "
        "checked at the CHAIN level (sufficient_context=False), not raw retrieval, "
        "but we still log what gets retrieved to watch for topic drift as techniques "
        "get more aggressive (e.g. MultiQuery casting a wider net).",
    ),
]
