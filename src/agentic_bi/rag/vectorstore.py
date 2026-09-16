"""FAISS vector store: build, persist, and load.

Building the index is expensive (one embedding API call per chunk) and
your documents change rarely -- so this deliberately separates "build and
save once" from "load from disk," rather than re-embedding on every app
startup. This distinction (build-time vs. query-time cost) is the same
one that matters at production scale, just visible here at toy scale.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agentic_bi.config.settings import get_settings
from agentic_bi.rag.embeddings import get_embeddings


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Embeds all chunks and builds a fresh in-memory FAISS index."""
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS) -> None:
    settings = get_settings()
    index_dir = Path(settings.faiss_index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def load_vectorstore() -> FAISS:
    """Loads a previously built index from disk.

    allow_dangerous_deserialization=True is required because FAISS's
    local save format uses pickle for the docstore. This is safe ONLY
    because you control what gets written to this path -- never load a
    FAISS index from an untrusted source with this flag set, since it's
    equivalent to unpickling arbitrary data.
    """
    settings = get_settings()
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(settings.faiss_index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_and_save_knowledge_base(chunks: list[Document]) -> FAISS:
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)
    return vectorstore


def get_mmr_retriever(vectorstore: FAISS, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5):
    """MMR retriever: balances relevance against diversity among selected chunks.

    fetch_k: how many candidates to consider before re-ranking (must be >= k).
    lambda_mult: 1.0 = pure relevance (behaves like plain similarity search),
                 0.0 = pure diversity (ignores query relevance almost entirely).
    0.5 is the library default and a reasonable starting point -- tune only
    after seeing where it under/over-corrects on the eval set.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def get_multi_query_retriever(vectorstore: FAISS, k: int = 4):
    """Wraps the base retriever with LLM-generated query variants.

    Under the hood: takes the input question, asks the configured LLM to
    generate 3 alternative phrasings (LangChain's built-in default prompt),
    runs similarity_search once per phrasing (4 total: original variants,
    not the original question itself unless the LLM happens to repeat it),
    then deduplicates the union by chunk identity before returning.

    This means each call to this retriever costs one extra LLM round-trip
    compared to plain similarity search -- worth remembering when this
    shows up as added latency/cost in Phase 27.
    """
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    from agentic_bi.llm.client import get_llm

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=get_llm())
