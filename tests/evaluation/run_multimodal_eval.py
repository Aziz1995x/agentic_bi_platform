"""
run_multimodal_eval.py
──────────────────────
Evaluates the multimodal retrieval pipeline against the fixed 10-case
eval set using the same 6 IR metrics established in Phase 5.

Metrics (identical definitions to the text eval harness)
─────────────────────────────────────────────────────────
  Hit Rate@K    — was at least one expected source retrieved?
  Recall@K      — fraction of expected sources retrieved
  Precision@K   — fraction of retrieved results that are expected sources
  Full Coverage — were ALL expected sources retrieved? (pass/fail)
  MRR@K         — reciprocal rank of the first relevant result
  NDCG@K        — normalised discounted cumulative gain (rank-weighted recall)

All metrics are macro-averaged across cases.

Key design constraints carried from Phase 5
────────────────────────────────────────────
- k is fixed for the entire run — all cases evaluated at the same k.
  Do not compare a run at k=4 against a different run at k=2.
- Ground truth is file_name, matched against result.file_name.
  The FAISS index stores file_name in metadata; the retriever exposes it
  on MultimodalResult.file_name.
- PDF expected sources match on file_name of the PDF, not individual
  page documents.  A PDF with 5 pages produces 5 Documents in the index,
  all with file_name="P01_q2_2018_business_report.pdf".  A hit on ANY
  page counts as the PDF being retrieved.

Usage
─────
    # From project root:
    python tests/evaluation/run_multimodal_eval.py

    # Custom k:
    python tests/evaluation/run_multimodal_eval.py --k 6

    # Compare similarity vs MMR:
    python tests/evaluation/run_multimodal_eval.py --method mmr

    # Save results to JSON:
    python tests/evaluation/run_multimodal_eval.py --output results/multimodal_eval.json
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.evaluation.multimodal_retrieval_cases import MULTIMODAL_CASES, MultimodalCase

logger = logging.getLogger(__name__)


# ── Metric computation ────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id:       str
    query:         str
    expected:      list[str]
    retrieved:     list[str]          # file_names in rank order
    hit_rate:      float              # 1.0 or 0.0
    recall:        float              # 0.0–1.0
    precision:     float              # 0.0–1.0
    full_coverage: float              # 1.0 or 0.0
    mrr:           float              # 0.0–1.0
    ndcg:          float              # 0.0–1.0


@dataclass
class EvalSummary:
    k:             int
    method:        str
    n_cases:       int
    hit_rate:      float
    recall:        float
    precision:     float
    full_coverage: float
    mrr:           float
    ndcg:          float
    case_results:  list[CaseResult]


def _is_hit(file_name: str, expected: list[str]) -> bool:
    """
    Match a retrieved file_name against the expected sources list.

    PDF pages: result file_name is the PDF filename (e.g.
    "P02_revenue_decline_analysis.pdf").  Expected source is also the PDF
    filename.  Direct equality check works for both images and PDFs.
    """
    return file_name in expected


def compute_metrics(
    retrieved_names: list[str],   # file_names in rank order, len = k
    expected: list[str],          # ground-truth file_names
    k: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Returns (hit_rate, recall, precision, full_coverage, mrr, ndcg).
    All inputs are already windowed to k.
    """
    top_k = retrieved_names[:k]
    hits = [name for name in top_k if _is_hit(name, expected)]

    # Hit Rate@K
    hit_rate = 1.0 if hits else 0.0

    # Recall@K
    recall = len(hits) / len(expected) if expected else 0.0

    # Precision@K
    precision = len(hits) / k if k > 0 else 0.0

    # Full Coverage
    full_coverage = 1.0 if all(e in top_k for e in expected) else 0.0

    # MRR@K — reciprocal rank of FIRST relevant result within top_k
    mrr = 0.0
    for rank, name in enumerate(top_k, start=1):
        if _is_hit(name, expected):
            mrr = 1.0 / rank
            break

    # NDCG@K
    # Relevance labels: 1 if the file_name is in expected, else 0.
    # DCG = sum(rel_i / log2(i+1)) for i in 1..k
    # Ideal DCG = DCG of a perfect ranking (all relevant docs first)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, name in enumerate(top_k, start=1)
        if _is_hit(name, expected)
    )
    # Ideal: place all len(expected) relevant docs at ranks 1..n_rel
    n_rel = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_rel + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return hit_rate, recall, precision, full_coverage, mrr, ndcg


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_eval(
    k: int = 4,
    method: str = "similarity",
    verbose: bool = True,
) -> EvalSummary:
    from agentic_bi.rag.multimodal.ingest import load_multimodal_vectorstore
    from agentic_bi.rag.multimodal.retriever import MultimodalRetriever

    vs        = load_multimodal_vectorstore()
    retriever = MultimodalRetriever(vs)

    case_results: list[CaseResult] = []

    for case in MULTIMODAL_CASES:
        results = retriever.retrieve(case.query, k=k, method=method)
        retrieved_names = [r.file_name for r in results]

        hit_rate, recall, precision, full_coverage, mrr, ndcg = compute_metrics(
            retrieved_names, case.expected_sources, k
        )

        cr = CaseResult(
            case_id=case.id,
            query=case.query,
            expected=case.expected_sources,
            retrieved=retrieved_names,
            hit_rate=hit_rate,
            recall=recall,
            precision=precision,
            full_coverage=full_coverage,
            mrr=mrr,
            ndcg=ndcg,
        )
        case_results.append(cr)

        if verbose:
            status = "PASS" if full_coverage == 1.0 else "FAIL"
            print(
                f"  [{status}] {case.id}  "
                f"HR={hit_rate:.1f}  R={recall:.3f}  "
                f"P={precision:.3f}  FC={full_coverage:.1f}  "
                f"MRR={mrr:.3f}  NDCG={ndcg:.3f}"
            )
            if full_coverage < 1.0:
                missing = [e for e in case.expected_sources if e not in retrieved_names]
                print(f"         missing: {missing}")
                print(f"         got:     {retrieved_names}")

    n = len(case_results)

    def _avg(attr: str) -> float:
        return sum(getattr(r, attr) for r in case_results) / n

    summary = EvalSummary(
        k=k,
        method=method,
        n_cases=n,
        hit_rate=_avg("hit_rate"),
        recall=_avg("recall"),
        precision=_avg("precision"),
        full_coverage=_avg("full_coverage"),
        mrr=_avg("mrr"),
        ndcg=_avg("ndcg"),
        case_results=case_results,
    )

    if verbose:
        _print_summary(summary)

    return summary


def _print_summary(s: EvalSummary) -> None:
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  Multimodal Retrieval Eval  |  k={s.k}  method={s.method}")
    print(bar)
    print(f"  Cases          : {s.n_cases}")
    print(f"  Hit Rate@{s.k:<3}   : {s.hit_rate:.3f}")
    print(f"  Recall@{s.k:<3}     : {s.recall:.3f}")
    print(f"  Precision@{s.k:<3}  : {s.precision:.3f}")
    print(f"  Full Coverage  : {s.full_coverage:.3f}   ← pass/fail")
    print(f"  MRR@{s.k:<3}        : {s.mrr:.3f}")
    print(f"  NDCG@{s.k:<3}       : {s.ndcg:.3f}")
    print(bar)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal retrieval evaluation")
    p.add_argument("--k",      type=int, default=4)
    p.add_argument("--method", choices=["similarity", "mmr"], default="similarity")
    p.add_argument("--output", type=Path, default=None,
                   help="Save full results to a JSON file")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress per-case output, print summary only")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,   # suppress library noise during eval
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    print(f"\nMultimodal retrieval eval  k={args.k}  method={args.method}")
    print("─" * 52)

    summary = run_eval(k=args.k, method=args.method, verbose=not args.quiet)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k":            summary.k,
            "method":       summary.method,
            "n_cases":      summary.n_cases,
            "hit_rate":     summary.hit_rate,
            "recall":       summary.recall,
            "precision":    summary.precision,
            "full_coverage": summary.full_coverage,
            "mrr":          summary.mrr,
            "ndcg":         summary.ndcg,
            "cases":        [asdict(cr) for cr in summary.case_results],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
