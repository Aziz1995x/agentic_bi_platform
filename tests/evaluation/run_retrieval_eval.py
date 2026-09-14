"""Runs the fixed retrieval test set against a given retriever and reports
recall + rank per case.

Usage:
    python -m tests.evaluation.run_retrieval_eval

Import run_retrieval_eval() directly to compare across retriever variants
(basic similarity, MMR, MultiQuery, etc.) in the same process.
"""

from dataclasses import dataclass
import math

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


@dataclass
class RetrievalMetrics:
    """Aggregate source-level retrieval metrics for the evaluation set.

    The current test set labels relevance at SOURCE level, not chunk level.
    Therefore Recall@K, Precision@K, Hit Rate@K, MRR and NDCG below are
    source-level metrics: repeated chunks from the same source are collapsed
    to their first occurrence before calculating the metrics.

    True chunk-level retrieval metrics require chunk-level relevance labels.
    True context precision/recall require answer/claim-level annotations
    (or an evaluation framework that derives them from a reference answer).
    """
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    full_source_coverage: float


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


def _unique_sources_in_order(sources: list[str]) -> list[str]:
    """Collapse repeated chunks from the same source, preserving first rank."""
    return list(dict.fromkeys(sources))


def _case_metrics(result: CaseResult, k: int) -> dict[str, float]:
    """Calculate source-level metrics for one case.

    All five metrics below (recall, precision, hit rate, reciprocal rank,
    NDCG) are now windowed to the same top_k -- a source appearing beyond
    rank k counts as "not retrieved" for every metric consistently. This
    was previously inconsistent: reciprocal_rank searched the full
    unique_retrieved list rather than top_k, meaning MRR could award
    credit for a source Hit Rate@K and Recall@K both correctly treated as
    absent. Restricting it here makes this genuinely "MRR@K," matching
    the rest of the report.
    """
    if not result.case.expected_sources:
        return {}

    expected = result.case.expected_sources
    unique_retrieved = _unique_sources_in_order(result.retrieved_sources_in_order)
    top_k = unique_retrieved[:k]

    relevant_in_top_k = len(expected.intersection(top_k))

    recall_at_k = relevant_in_top_k / len(expected)
    # Precision@K here divides by however many unique sources actually came
    # back within the window (len(top_k)), not always by the nominal k --
    # e.g. if a retriever returns only 2 unique sources when k=4, precision
    # is computed as "relevant / 2", not "relevant / 4". This is a legitimate
    # alternate definition (precision among what was actually retrieved,
    # not among a hypothetical fixed-size slate) -- worth knowing if
    # cross-checking against a tool that always divides by k.
    precision_at_k = relevant_in_top_k / len(top_k) if top_k else 0.0
    hit_rate_at_k = 1.0 if relevant_in_top_k > 0 else 0.0

    # Reciprocal rank of the first relevant source WITHIN the top-k window
    # (previously searched the full unique_retrieved list -- fixed so this
    # is genuine MRR@K, consistent with every other metric here).
    first_relevant_rank = next(
        (rank for rank, source in enumerate(top_k, start=1)
         if source in expected),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    # Binary-relevance NDCG@K at source level.
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(top_k, start=1)
        if source in expected
    )

    ideal_relevant = min(len(expected), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    ndcg_at_k = dcg / idcg if idcg else 0.0

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "hit_rate_at_k": hit_rate_at_k,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": ndcg_at_k,
    }



def calculate_metrics(results: list[CaseResult], k: int) -> RetrievalMetrics:
    """Aggregate source-level retrieval metrics across real test cases."""
    real_cases = [r for r in results if r.case.expected_sources]

    if not real_cases:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    case_metrics = [_case_metrics(r, k) for r in real_cases]

    return RetrievalMetrics(
        recall_at_k=sum(m["recall_at_k"] for m in case_metrics) / len(case_metrics),
        precision_at_k=sum(m["precision_at_k"] for m in case_metrics) / len(case_metrics),
        hit_rate_at_k=sum(m["hit_rate_at_k"] for m in case_metrics) / len(case_metrics),
        mrr=sum(m["reciprocal_rank"] for m in case_metrics) / len(case_metrics),
        ndcg_at_k=sum(m["ndcg_at_k"] for m in case_metrics) / len(case_metrics),
        full_source_coverage=sum(r.hit for r in real_cases) / len(real_cases),
    )


def run_retrieval_eval(
    retriever: BaseRetriever,
    label: str = "retriever",
    k: int | None = None,
) -> list[CaseResult]:
    results = [evaluate_case(retriever, case) for case in RETRIEVAL_TEST_CASES]

    # Infer K from the number of returned chunks when the caller doesn't
    # provide it. This matches the actual retrieval depth for each case.
    if k is None:
        k = max((len(r.retrieved_sources_in_order) for r in results), default=0)

    print_report(results, label, k=k)
    return results


def print_report(results: list[CaseResult], label: str, k: int) -> None:
    print(f"\n{'=' * 70}\nRetrieval Quality Report: {label}\n{'=' * 70}")

    real_cases = [r for r in results if r.case.expected_sources]
    hits = sum(1 for r in real_cases if r.hit)
    metrics = calculate_metrics(results, k)

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

            case_metric = _case_metrics(r, k)
            print(
                f"    metrics:   "
                f"Recall@{k}={case_metric['recall_at_k']:.2f}, "
                f"Precision@{k}={case_metric['precision_at_k']:.2f}, "
                f"HitRate@{k}={case_metric['hit_rate_at_k']:.2f}, "
                f"RR={case_metric['reciprocal_rank']:.2f}, "
                f"NDCG@{k}={case_metric['ndcg_at_k']:.2f}"
            )

        print(f"    note: {c.note}")

    print(f"\n{'-' * 70}")
    print("SOURCE-LEVEL AGGREGATE METRICS")
    print(f"Recall@{k}:     {metrics.recall_at_k:.3f}")
    print(f"Precision@{k}:  {metrics.precision_at_k:.3f}")
    print(f"Hit Rate@{k}:   {metrics.hit_rate_at_k:.3f}")
    print(f"MRR@{k}:            {metrics.mrr:.3f}")
    print(f"NDCG@{k}:       {metrics.ndcg_at_k:.3f}")
    print(f"Full Coverage:  {metrics.full_source_coverage:.3f}")
    print(
        f"\nFull Coverage: {hits}/{len(real_cases)} cases fully retrieved "
        f"({len(results) - len(real_cases)} negative control(s) excluded from score)"
    )
    print(
        "\nNote: These metrics use expected_sources, so they are SOURCE-level "
        "metrics. True chunk-level Recall/Precision and true Context "
        "Precision/Recall need chunk/claim-level relevance labels."
    )
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
        keys = doc.metadata.keys()
        for key in keys:
            if key != 'source':
                print(f"      {key}: {doc.metadata[key]}")
        print(f"      content: {content_preview}{'...' if len(doc.page_content) > max_chars else ''}")
    print(f"{'=' * 70}\n")



def run_colbert_eval(RAG, label: str, k: int = 4) -> None:
    """Evaluate ColBERT using the same source-level metrics.

    ColBERT doesn't implement BaseRetriever.invoke() in this harness, so it
    needs its own runner. The metric definitions are otherwise identical.
    """
    from tests.evaluation.retrieval_cases import RETRIEVAL_TEST_CASES
    from agentic_bi.rag.colbert_retriever import colbert_search

    print(f"\n{'=' * 70}\nRetrieval Quality Report: {label}\n{'=' * 70}")

    results = []

    for case in RETRIEVAL_TEST_CASES:
        docs = colbert_search(RAG, case.question, k=k)
        retrieved_sources = [d.metadata.get("source", "UNKNOWN") for d in docs]

        best_rank = {}
        for rank, source in enumerate(retrieved_sources, start=1):
            if source not in best_rank:
                best_rank[source] = rank

        missing = case.expected_sources - set(retrieved_sources)
        hit = len(missing) == 0

        results.append(
            CaseResult(
                case=case,
                retrieved_sources_in_order=retrieved_sources,
                hit=hit,
                missing=missing,
                best_rank=best_rank,
            )
        )

        status = (
            "PASS"
            if hit
            else "FAIL"
            if case.expected_sources
            else "  (negative control)"
        )
        print(f"\n[{case.id}] {status} -- {case.question}")
        print(f"    expected:  {sorted(case.expected_sources) or '(none)'}")
        print(f"    retrieved: {retrieved_sources}")
        if missing:
            print(f"    MISSING:   {sorted(missing)}")

    metrics = calculate_metrics(results, k=k)

    print(f"\n{'-' * 70}")
    print("SOURCE-LEVEL AGGREGATE METRICS")
    print(f"Recall@{k}:     {metrics.recall_at_k:.3f}")
    print(f"Precision@{k}:  {metrics.precision_at_k:.3f}")
    print(f"Hit Rate@{k}:   {metrics.hit_rate_at_k:.3f}")
    print(f"MRR@{k}:            {metrics.mrr:.3f}")
    print(f"NDCG@{k}:       {metrics.ndcg_at_k:.3f}")
    print(f"Full Coverage:  {metrics.full_source_coverage:.3f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    from agentic_bi.rag.vectorstore import get_mmr_retriever, get_multi_query_retriever
    from agentic_bi.rag.reranker import get_reranking_retriever
    from agentic_bi.rag.parent_retriever import build_parent_document_retriever
    from agentic_bi.rag.loaders import load_knowledge_base_documents
    from agentic_bi.rag.bm25_retriever import build_hybrid_retriever, build_bm25_retriever
    from agentic_bi.rag.chunking import split_documents
    from agentic_bi.rag.contextual_chunking import contextualize_chunks
    from agentic_bi.rag.raptor import build_raptor_tree

    vectorstore = load_vectorstore()
    raw_docs = load_knowledge_base_documents()
    chunks = split_documents(documents=raw_docs)

    # basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # mmr_retriever = get_mmr_retriever(vectorstore, k=4, fetch_k=10, lambda_mult=0.8)
    # mq_retriever = get_multi_query_retriever(vectorstore=vectorstore, k=4)
    # rerank_retriever = get_reranking_retriever(vectorstore=vectorstore, fetch_k=10, top_n=4)
    # parent_retriever, _ = build_parent_document_retriever(raw_documents=raw_docs, k=4)
    # bm25_retriever = build_bm25_retriever(chunks=chunks, k=2)
    # hybrid_retriever = build_hybrid_retriever(vectorstore=vectorstore,
    #                                           chunks=chunks,
    #                                           k=2,
    #                                           dense_weight=0.5)

    # CONTEXTUAL RETRIEVER
    # from langchain_community.vectorstores import FAISS
    # from agentic_bi.rag.embeddings import get_embeddings
    # print(f"Contextualizing {len(chunks)} chunks (one LLM call each)...")
    # contextual_chunks = contextualize_chunks(raw_docs, chunks)
    # contextual_vectorstore = FAISS.from_documents(contextual_chunks, get_embeddings())
    # contextual_retriever = contextual_vectorstore.as_retriever(search_kwargs={"k": 4})

    # RAPTOR ALGORITHM
    # raptor_nodes = build_raptor_tree(chunks, max_levels=2)
    # print(f"RAPTOR tree: {len(chunks)} leaves + {len(raptor_nodes) - len(chunks)} summary nodes "
    #       f"= {len(raptor_nodes)} total")
    # raptor_vectorstore = FAISS.from_documents(raptor_nodes, get_embeddings())
    # raptor_retriever = raptor_vectorstore.as_retriever(search_kwargs={"k": 4})

    # ENRICHED RETRIEVER (much like self query retriever with enriched metadata)
    # from agentic_bi.rag.metadata_enrichment import enrich_chunk_metadata
    # metadata_enriched_chunks = enrich_chunk_metadata(chunks=chunks, raw_documents=raw_docs)
    # enriched_vectorstore = FAISS.from_documents(metadata_enriched_chunks, get_embeddings())
    # enriched_retriever = enriched_vectorstore.as_retriever(search_kwargs={"k": 4})

    # SELF QUERY RETRIEVER (DID NOT WORK DUE TO VERSION MISMATCH IN MODULES)
    # SelfQueryRetriever logs its generated filter + rewritten query via
    # langchain.retrievers.self_query.base at INFO level -- same pattern
    # as MultiQueryRetriever back in Step 2. This is the actual thing
    # worth reading, not just the recall number.
    # import logging
    # from agentic_bi.rag.self_query_retriever import build_self_query_retriever
    # logging.basicConfig()
    # logging.getLogger("langchain.retrievers.self_query.base").setLevel(logging.INFO)
    # logging.getLogger("langchain.chains.query_constructor.base").setLevel(logging.INFO)
    # self_query_retriever = build_self_query_retriever(enriched_vectorstore, k=4)

    # Colbert Retriever: Not Working because of env version issues!
    # from src.agentic_bi.rag.colbert_retriever import build_colbert_index
    # raw_docs = load_knowledge_base_documents()
    # chunks = split_documents(raw_docs)
    # COLBERT_RAG, _ = build_colbert_index(chunks=chunks)

    # run_retrieval_eval(basic_retriever, label="Basic similarity search (k=4)")
    # run_retrieval_eval(mmr_retriever, label="MMR (k=4, fetch_k=10, lambda=0.8)")
    # run_retrieval_eval(mq_retriever, label="MQR (k=4)")
    # run_retrieval_eval(rerank_retriever, label="Reranked (fetch_k=10, top_n=4)")
    # run_retrieval_eval(parent_retriever, label="ParentDocumentRetriever (child=250, parent=1200, k=4)")
    # run_retrieval_eval(bm25_retriever, label="bm25_retriever (k=2)")
    # run_retrieval_eval(hybrid_retriever, label="hybrid_retriever (k=2, dense_weight=0.5)")
    # run_retrieval_eval(contextual_retriever, label="Contextual Retrieval (k=4)")
    # run_retrieval_eval(raptor_retriever, label="RAPTOR (k=4)")
    # run_retrieval_eval(enriched_retriever, label="Enriched Retriever (k=4)")
    # run_retrieval_eval(self_query_retriever, label="Self-Query (k=4)")        # did not work
    # run_colbert_eval(COLBERT_RAG, label="ColBERT (k=4)", k=4)     # did not work

    # Content-level drill-down on the flagged cases only -- cheap to run,
    # and the only way to tell whether source-level "PASS" is hiding a
    # real quality difference between the two retrievers.
    # flagged_case_ids = {4, 6, 7, 8, 9}
    # flagged_cases = [c for c in RETRIEVAL_TEST_CASES if c.id in flagged_case_ids]
    # for case in flagged_cases:
    #     print("\n########## BASIC ##########")
    #     inspect_case_content(basic_retriever, case)
        # print("########## MMR ##########")
        # inspect_case_content(mmr_retriever, case)
        # print("########## MQR ##########")
        # inspect_case_content(mq_retriever, case)
        # print("########## RERANKING - ContextualCompressionRetriever ##########")
        # inspect_case_content(rerank_retriever, case)
        # print("########## PARENT DOCUMENT RETRIEVER ##########")
        # inspect_case_content(parent_retriever, case)
        # print("########## BM25 DOCUMENT RETRIEVER ##########")
        # inspect_case_content(bm25_retriever, case)
        # print("########## HYBRID (DENSE+BM25) DOCUMENT RETRIEVER ##########")
        # inspect_case_content(hybrid_retriever, case)
        # print("########## Contextual Query DOCUMENT RETRIEVER ##########")
        # inspect_case_content(contextual_retriever, case)
        # print("########## METADATA ENRICHED DOCUMENT RETRIEVER ##########")
        # inspect_case_content(enriched_retriever, case)

    from agentic_bi.rag.composed_retriever import build_composed_retriever

    for label_k, k in [("k=4", 4), ("k=2", 2)]:
        basic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        composed_retriever = build_composed_retriever(
            vectorstore, chunks, fetch_k=10, top_n=k, dense_weight=0.5
        )
        run_retrieval_eval(basic_retriever, label=f"Basic ({label_k})", k=k)

        flagged_case_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        flagged_cases = [c for c in RETRIEVAL_TEST_CASES if c.id in flagged_case_ids]

        for case in flagged_cases:
            print("\n########## BASIC ##########")
            inspect_case_content(basic_retriever, case)

        run_retrieval_eval(composed_retriever, label=f"Composed: Hybrid->Rerank ({label_k})", k=k)
        for case in flagged_cases:
            print("\n########## COMPOSED ##########")
            inspect_case_content(composed_retriever, case)