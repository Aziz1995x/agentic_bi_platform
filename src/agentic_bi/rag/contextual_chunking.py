"""Contextual Retrieval: prepend an LLM-generated situating description
to each chunk before embedding.

Cost note, worth internalizing: this makes ONE LLM call PER CHUNK at
INDEX-BUILD time, not at query time (unlike MultiQuery's per-QUERY LLM
call). This is a one-time cost when the knowledge base is built or
rebuilt, not a recurring per-question cost -- a very different tradeoff
profile than the other techniques tested so far, and worth naming
explicitly in the eventual verdict, since "expensive but only once" reads
very differently than "expensive on every query."
"""

from langchain_core.documents import Document

from agentic_bi.llm.client import get_llm
from agentic_bi.prompts.contextual_retrieval import contextual_retrieval_prompt


def contextualize_chunks(
    raw_documents: list[Document],
    chunks: list[Document],
) -> list[Document]:
    """For each chunk, finds its parent raw document (matched by source
    filename) and asks the LLM to generate a situating description,
    which is prepended to the chunk's page_content.

    Original chunk content is preserved in metadata (original_content)
    so you can inspect/compare before vs. after contextualization.
    """
    llm = get_llm()
    chain = contextual_retrieval_prompt | llm

    # Build a quick lookup: source filename -> full document text
    doc_lookup = {doc.metadata["source"]: doc.page_content for doc in raw_documents}

    contextualized_chunks = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        full_document = doc_lookup.get(source, "")

        context_description = chain.invoke(
            {"document": full_document, "chunk": chunk.page_content}
        ).content

        new_content = f"{context_description}\n\n{chunk.page_content}"
        new_metadata = {**chunk.metadata, "original_content": chunk.page_content}

        contextualized_chunks.append(
            Document(page_content=new_content, metadata=new_metadata)
        )

    return contextualized_chunks
