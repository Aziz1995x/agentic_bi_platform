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
