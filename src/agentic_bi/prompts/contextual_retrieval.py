"""Prompt for Anthropic's Contextual Retrieval technique.

Given a full document and one chunk from it, ask the LLM to produce a
short situating description -- this gets PREPENDED to the chunk before
embedding, so the embedding itself carries context the isolated chunk
text alone would lose.
"""

from langchain_core.prompts import ChatPromptTemplate

CONTEXTUAL_RETRIEVAL_SYSTEM_PROMPT = """You are helping prepare a document
chunk for a retrieval system. You will be given the full document and one
chunk from it. Write a short (1-2 sentence) description that situates this
chunk within the document -- what section it's from, and what broader rule
or scope it depends on or relates to. Be concise and factual. Do not
summarize the chunk's own content -- only add the surrounding context
needed to understand it in isolation."""

contextual_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUAL_RETRIEVAL_SYSTEM_PROMPT),
        (
            "human",
            "Full document:\n{document}\n\n"
            "Chunk to contextualize:\n{chunk}\n\n"
            "Provide only the situating description, nothing else.",
        ),
    ]
)
