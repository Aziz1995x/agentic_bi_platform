"""RAG answer-generation chain: retriever -> format -> prompt -> LLM.

Chain of custody:

    question (str)
        |
    retriever.invoke(question)        -> list[Document]
        |
    format_docs()                     -> str (context block)
        |
    rag_answer_prompt.invoke(...)     -> PromptValue
        |
    structured_llm                    -> RAGAnswer

RunnablePassthrough carries the original question through unchanged
alongside the retrieval step, exactly like the Phase 3 RoutedQuestion
pattern -- the LLM needs both the formatted context AND the raw question
in the same prompt call.
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from agentic_bi.llm.client import get_llm
from agentic_bi.prompts.rag import rag_answer_prompt
from agentic_bi.rag.schemas import RAGAnswer
from agentic_bi.rag.vectorstore import load_vectorstore


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved chunks into a single context string, each
    tagged with its source so the LLM can cite accurately."""
    return "\n\n".join(f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs)


def build_rag_chain(retriever: VectorStoreRetriever | None = None):
    """Builds the full RAG chain. Accepts an optional retriever so tests
    (and later, Phase 5 comparisons) can swap in different retrieval
    strategies without touching this function."""
    if retriever is None:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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


def ask_knowledge_base(question: str) -> RAGAnswer:
    chain = build_rag_chain()
    return chain.invoke(question)
