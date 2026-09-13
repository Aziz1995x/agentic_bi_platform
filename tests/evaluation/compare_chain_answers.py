"""Compares full RAG chain answers (not just retrieval) across retriever
strategies for a specific question. This is the higher-signal comparison
-- retrieval recall is a proxy for answer quality, this checks the thing
that actually matters.
"""

from agentic_bi.llm.client import get_llm
from agentic_bi.prompts.rag import rag_answer_prompt
from agentic_bi.rag.chain import format_docs
from agentic_bi.rag.schemas import RAGAnswer
from agentic_bi.rag.vectorstore import get_mmr_retriever, load_vectorstore, get_multi_query_retriever
from agentic_bi.rag.reranker import get_reranking_retriever
from agentic_bi.rag.loaders import load_knowledge_base_documents
from agentic_bi.rag.parent_retriever import build_parent_document_retriever
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

def compare_answers_basic_vs_mqr(question: str, k: int = 4) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    mq_retriever = get_multi_query_retriever(vectorstore=vectorstore, k=4)

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"MQR (lambda={k})", mq_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()

def compare_answers_basic_vs_rerank(question: str) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rerank_retriever = get_reranking_retriever(vectorstore=vectorstore, fetch_k=10, top_n=4)

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"RERANK", rerank_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()

def compare_answers_basic_vs_parent_doc(question: str) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    raw_docs = load_knowledge_base_documents()
    parent_retriever, _ = build_parent_document_retriever(raw_documents=raw_docs, k=4)

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"PARENT_RETRIEVER", parent_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()

def compare_answers_basic_vs_hybrid_doc(question: str) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    from agentic_bi.rag.chunking import split_documents
    from agentic_bi.rag.bm25_retriever import build_hybrid_retriever

    raw_docs = load_knowledge_base_documents()
    chunks = split_documents(documents=raw_docs)
    hybrid_retriever = build_hybrid_retriever(vectorstore=vectorstore,
                                              chunks=chunks,
                                              k=2,
                                              dense_weight=0.5)

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"HYBRID_RETRIEVER (k=2)", hybrid_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()


def compare_answers_basic_vs_contextual_doc(question: str) -> None:
    vectorstore = load_vectorstore()
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    from langchain_community.vectorstores import FAISS
    from agentic_bi.rag.embeddings import get_embeddings
    from agentic_bi.rag.chunking import split_documents
    from agentic_bi.rag.contextual_chunking import contextualize_chunks

    raw_docs = load_knowledge_base_documents()
    chunks = split_documents(documents=raw_docs)
    contextual_chunks = contextualize_chunks(raw_docs, chunks)
    contextual_vectorstore = FAISS.from_documents(contextual_chunks, get_embeddings())
    contextual_retriever = contextual_vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"\nQUESTION: {question}\n")

    for label, retriever in [("BASIC", basic_retriever), (f"Contextual Retrieval (k=4)", contextual_retriever)]:
        chain = build_chain_with_retriever(retriever)
        result: RAGAnswer = chain.invoke(question)
        print(f"--- {label} ---")
        print(f"answer: {result.answer}")
        print(f"sufficient_context: {result.sufficient_context}")
        print(f"citations: {[(c.source, c.excerpt) for c in result.citations]}")
        print()


if __name__ == "__main__":
    flagged_case_ids = {4,5,6,7,8,9}
    flagged_cases = [c for c in RETRIEVAL_TEST_CASES if c.id in flagged_case_ids]
    for case in flagged_cases:
        # compare_answers(case.question, lambda_mult=0.8)
        # compare_answers_basic_vs_rerank(case.question)
        # compare_answers_basic_vs_parent_doc(case.question)
        # compare_answers_basic_vs_hybrid_doc(question=case.question)
        compare_answers_basic_vs_contextual_doc(question=case.question)
