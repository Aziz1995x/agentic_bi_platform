"""Splits loaded documents into retrieval-sized chunks.

Chunk size and overlap are the two knobs that most directly affect
retrieval quality, and there's no universally correct value -- it depends
on how self-contained your document's sections are. Your policy docs use
markdown headers (##) to separate distinct rules (e.g. "Restocking Fee" is
its own section) -- a good chunk size keeps one header's content together
rather than splitting mid-rule.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Starting values -- these get measured, not assumed, in Phase 5 when you
# compare retrieval quality across configurations.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents using recursive character splitting.

    RecursiveCharacterTextSplitter tries a list of separators in order
    (default: ["\\n\\n", "\\n", " ", ""]) -- it prefers splitting on
    paragraph breaks, falling back to sentences/words only if a chunk is
    still too big. This is why it tends to respect markdown structure
    reasonably well even without a markdown-aware splitter, though not
    perfectly -- worth checking chunk boundaries against your actual
    section headers below.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
