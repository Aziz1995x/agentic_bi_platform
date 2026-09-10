"""Runs the fixed retrieval test set against a given retriever and reports
recall + rank per case.

Usage:
    python -m tests.evaluation.run_retrieval_eval

Import run_retrieval_eval() directly to compare across retriever variants
(basic similarity, MMR, MultiQuery, etc.) in the same process.
"""

from dataclasses import dataclass

from langchain_core.retrievers import BaseRetriever

from agentic_bi.rag.vectorstore import load_vectorstore
from tests.evaluation.retrieval_cases import RETRIEVAL_TEST_CASES, RetrievalTestCase


@dataclass
class CaseResult:
    case: RetrievalTestCase
    retrieved_sources_in_order: list[str]
    hit: bool          # all expected_sources present in retrieved set (or, for negative
    # controls, always True -- see note in report())
    missing: set[str]  # expected sources NOT retrieved
    best_rank: dict[str, int]  # source -> 1-indexed rank of its first appearance


def evaluate_case(retriever: BaseRetriever, case: RetrievalTestCase) -> CaseResult:
    docs = retriever.invoke(case.question)
    retrieved_sources = [doc.metadata["source"] for doc in docs]

    best_rank: dict[str, int] = {}
    for rank, source in enumerate(retrieved_sources, start=1):
        if source not in best_rank:
            best_rank[source] = rank

    missing = case.expected_sources - set(retrieved_sources)
    hit = len(missing) == 0

    return CaseResult(
        case=case,
        retrieved_sources_in_order=retrieved_sources,
        hit=hit,
        missing=missing,
        best_rank=best_rank,
    )


def run_retrieval_eval(retriever: BaseRetriever, label: str = "retriever") -> list[CaseResult]:
    results = [evaluate_case(retriever, case) for case in RETRIEVAL_TEST_CASES]
    print_report(results, label)
    return results


def print_report(results: list[CaseResult], label: str) -> None:
    print(f"\n{'=' * 70}\nRetrieval Quality Report: {label}\n{'=' * 70}")

    real_cases = [r for r in results if r.case.expected_sources]
    hits = sum(1 for r in real_cases if r.hit)

    for r in results:
        c = r.case
        if not c.expected_sources:
            status = "  (negative control -- inspect manually)"
        else:
            status = "PASS" if r.hit else "FAIL"

        print(f"\n[{c.id}] {status} -- {c.question}")
        print(f"    expected:  {sorted(c.expected_sources) or '(none)'}")
        print(f"    retrieved: {r.retrieved_sources_in_order}")
        if r.missing:
            print(f"    MISSING:   {sorted(r.missing)}")

        # New: show where each expected source actually landed, not just
        # whether it showed up. A source at rank 1 and one at rank 4 both
        # count as "PASS" above, but they're very different retrieval
        # outcomes once k gets trimmed down or a validation/reranking step
        # only looks at the top 1-2 results.
        if c.expected_sources:
            rank_str = ", ".join(
                f"{src}: rank {r.best_rank.get(src, 'MISSING')}"
                for src in sorted(c.expected_sources)
            )
            print(f"    ranks:     {rank_str}")

        print(f"    note: {c.note}")

    print(f"\n{'-' * 70}")
    print(f"Recall: {hits}/{len(real_cases)} cases fully retrieved "
          f"({len(results) - len(real_cases)} negative control(s) excluded from score)")
    print(f"{'=' * 70}\n")


def inspect_case_content(retriever: BaseRetriever, case: RetrievalTestCase, max_chars: int = 300) -> None:
    """Prints full chunk content for one case, not just source filenames.

    Use this whenever source-level recall passes but you need to verify
    the SAME relevant section actually survived -- source-level recall
    can't distinguish "4 distinct relevant chunks" from "1 relevant chunk
    duplicated via overlap" or "half the relevant content lost to a
    diversity tradeoff."
    """
    docs = retriever.invoke(case.question)
    print(f"\n{'=' * 70}")
    print(f"CONTENT INSPECTION -- [{case.id}] {case.question}")
    print(f"{'=' * 70}")
    for i, doc in enumerate(docs, start=1):
        content_preview = doc.page_content[:max_chars].replace("\n", " ")
        print(f"\n  [{i}] source: {doc.metadata['source']}")
        print(f"      content: {content_preview}{'...' if len(doc.page_content) > max_chars else ''}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    from agentic_bi.rag.vectorstore import get_mmr_retriever, get_multi_query_retriever
    from agentic_bi.rag.reranker import get_reranking_retriever

    vectorstore = load_vectorstore()

    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    mmr_retriever = get_mmr_retriever(vectorstore, k=4, fetch_k=10, lambda_mult=0.8)
    mq_retriever = get_multi_query_retriever(vectorstore=vectorstore, k=4)
    rerank_retriever = get_reranking_retriever(vectorstore=vectorstore, fetch_k=10, top_n=2)

    run_retrieval_eval(basic_retriever, label="Basic similarity search (k=2)")
    # run_retrieval_eval(mmr_retriever, label="MMR (k=4, fetch_k=10, lambda=0.8)")
    # run_retrieval_eval(mq_retriever, label="MQR (k=4)")
    run_retrieval_eval(rerank_retriever, label="Reranked (fetch_k=10, top_n=2)")

    # Content-level drill-down on the flagged cases only -- cheap to run,
    # and the only way to tell whether source-level "PASS" is hiding a
    # real quality difference between the two retrievers.
    flagged_case_ids = {8,}
    flagged_cases = [c for c in RETRIEVAL_TEST_CASES if c.id in flagged_case_ids]

    for case in flagged_cases:
        print("\n########## BASIC ##########")
        inspect_case_content(basic_retriever, case)
        # print("########## MMR ##########")
        # inspect_case_content(mmr_retriever, case)
        # print("########## MQR ##########")
        # inspect_case_content(mq_retriever, case)
        print("########## RERANKING - ContextualCompressionRetriever ##########")
        inspect_case_content(rerank_retriever, case)
