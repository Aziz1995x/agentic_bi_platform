"""Document loading for the RAG knowledge base.

All source documents currently live as project-authored markdown files in
documents/. Using DirectoryLoader + TextLoader (rather than hand-rolling
file iteration) means adding a new policy doc later requires zero code
changes -- drop the .md file in the folder and it's picked up.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from agentic_bi.config.settings import get_settings


def load_knowledge_base_documents() -> list[Document]:
    """Loads every .md file under documents/ as a LangChain Document.

    Each Document gets a `source` metadata field (the filename) attached
    automatically by TextLoader -- this is what makes citation possible
    later: when the retriever returns a chunk, you can tell the user
    *which policy document* it came from.
    """
    settings = get_settings()
    docs_path = Path(settings.documents_dir)

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    # Normalize the source path to just the filename -- the full path is
    # an implementation detail the LLM/user shouldn't see in citations.
    for doc in documents:
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    return documents
