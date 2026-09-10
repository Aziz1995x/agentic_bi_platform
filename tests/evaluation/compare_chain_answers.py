"""Compares full RAG chain answers (not just retrieval) across retriever
strategies for a specific question. This is the higher-signal comparison
-- retrieval recall is a proxy for answer quality, this checks the thing
that actually matters.
"""

from agentic_bi.llm.client import get_llm
from agentic_bi.prompts.rag import rag_answer_prompt
from agentic_bi.rag.chain import format_docs
from agentic_bi.rag.schemas import RAGAnswer
from agentic_bi.rag.vectorstore import get_mmr_retriever, load_vectorstore
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from tests.evaluation.retrieval_cases import RETRIEVAL_TEST_CASES, RetrievalTestCase


def build_chain_with_retriever(retriever):
    llm = get_llm()
    structured_llm = llm.with_structured_output(RAGAnswer)
    return (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | rag_answer_prompt
        | structured_llm
    )


def compare_answers(question: str, lambda_mult: float = 0.5) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    mmr_retriever = get_mmr_retriever(vectorstore, k=4, fetch_k=10, lambda_mult=lambda_mult)

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"MMR (lambda={lambda_mult})", mmr_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()


if __name__ == "__main__":
    flagged_case_ids = {9, }
    flagged_cases = [c for c in RETRIEVAL_TEST_CASES if c.id in flagged_case_ids]
    for case in flagged_cases:
        compare_answers(case.question, lambda_mult=1)
