"""ParentDocumentRetriever: small chunks for search precision, larger
chunks for context completeness.

Two splitters, two roles:
  - child_splitter: small chunks (e.g. 250 chars) get embedded and indexed
    in the vectorstore -- similarity search operates on THESE, so a query
    matching one sentence inside a long section still finds it precisely.
  - parent_splitter: larger chunks (e.g. 1200 chars) define what actually
    gets RETURNED once a child chunk is matched -- the retriever looks up
    which parent a matched child belongs to and returns the whole parent,
    not just the small fragment that matched.

This requires a docstore (here: InMemoryStore) to hold the parent chunks
and a mapping from child ID -> parent ID, separate from the vectorstore,
which only ever holds child embeddings. Two storage systems, one retriever
-- this is worth remembering as an architectural pattern: LangChain's
"multi-vector" retrievers generally split "what's searched" from "what's
returned" this way.
"""

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_bi.rag.embeddings import get_embeddings


def build_parent_document_retriever(
    raw_documents: list[Document],
    child_chunk_size: int = 250,
    child_chunk_overlap: int = 30,
    parent_chunk_size: int = 1200,
    parent_chunk_overlap: int = 100,
    k: int = 4,
) -> tuple[ParentDocumentRetriever, VectorStore]:
    """Builds a ParentDocumentRetriever from raw (unsplit) documents.

    Note this takes RAW documents, not pre-chunked ones -- unlike your
    other retrievers, ParentDocumentRetriever does its OWN two-level
    splitting internally via add_documents(). Feeding it already-split
    800-char chunks would double-split them incorrectly.
    """
    from langchain_community.vectorstores import FAISS

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
    )

    embeddings = get_embeddings()
    # Start with an empty FAISS index -- ParentDocumentRetriever populates
    # it via add_documents() below, using child_splitter internally.
    empty_vectorstore = FAISS.from_texts(["placeholder"], embeddings)
    empty_vectorstore.delete([empty_vectorstore.index_to_docstore_id[0]])

    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=empty_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k},
    )
    retriever.add_documents(raw_documents)

    return retriever, empty_vectorstore
