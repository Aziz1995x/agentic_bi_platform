"""Structured output for RAG answers."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="The source document filename")
    excerpt: str = Field(
        description="The specific sentence or phrase from that source that supports the claim"
    )


class RAGAnswer(BaseModel):
    answer: str = Field(description="The synthesized answer to the question")
    citations: list[Citation] = Field(
        description="Sources used, one per distinct document referenced"
    )
    sufficient_context: bool = Field(
        description="False if the retrieved context did not contain enough information to fully answer"
    )
