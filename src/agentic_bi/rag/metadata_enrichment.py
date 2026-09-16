"""Enriches chunks with structured metadata derived from document
structure, so Self-Query Retrieval has real fields to filter on.

Metadata fields added, and why each one:
  - document_type: coarse category (policy | definitions | schema).
    Lets a query like "according to policy" filter out the data
    dictionary, or vice versa.
  - section_title: the nearest preceding markdown ## heading. Lets a
    query target a specific rule area (e.g. "Restocking Fee") even if
    the chunk's prose doesn't repeat the heading verbatim.
  - mentions_segment: whether the chunk's text references Enterprise
    and/or Retail segmentation, derived by simple keyword presence, not
    an LLM call -- deterministic and free, unlike Contextual Retrieval's
    per-chunk LLM cost. Lets a query like "for enterprise customers"
    filter to only chunks that actually discuss segmentation.

Deriving section_title from the chunk's own leading '## ' line (if
present) is a heuristic, not a parser -- a chunk that starts mid-section
(due to chunk_overlap) won't have a clean heading. Good enough for this
corpus's markdown structure; a production system with less consistent
source formatting would need a real markdown AST parser instead.
"""

import re

from langchain_core.documents import Document

_DOCUMENT_TYPE_MAP = {
    "kpi_definitions.md": "definitions",
    "refund_and_returns_policy.md": "policy",
    "pricing_and_freight_policy.md": "policy",
    "customer_segmentation.md": "policy",
    "data_dictionary.md": "schema",
}

_SEGMENT_KEYWORDS = {
    "enterprise": ["enterprise"],
    "retail": ["retail"],
}


def _extract_section_title(chunk_text: str) -> str:
    """Finds the last '## Heading' line at or before this chunk's start.
    Falls back to '(unspecified)' if the chunk starts mid-paragraph with
    no heading visible (common with chunk_overlap carrying content across
    a boundary)."""
    matches = re.findall(r"^##\s+(.+)$", chunk_text, flags=re.MULTILINE)
    return matches[0].strip() if matches else "(unspecified)"


def _detect_segments(chunk_text: str) -> list[str]:
    lowered = chunk_text.lower()
    return [
        segment
        for segment, keywords in _SEGMENT_KEYWORDS.items()
        if any(kw in lowered for kw in keywords)
    ]


def _build_section_map(full_document_text: str) -> list[tuple[int, str]]:
    """Returns a list of (char_offset, heading_text) for every '## '
    heading in the full document, in order. Used to look up which
    section governs any given chunk by its start offset -- correct
    even when a chunk starts mid-section with no heading of its own."""
    return [
        (m.start(), m.group(1).strip())
        for m in re.finditer(r"^##\s+(.+)$", full_document_text, flags=re.MULTILINE)
    ]


def _section_title_for_offset(section_map: list[tuple[int, str]], offset: int) -> str:
    """Finds the last heading at or before this character offset."""
    governing = "(unspecified)"
    for heading_offset, heading_text in section_map:
        if heading_offset <= offset:
            governing = heading_text
        else:
            break
    return governing


def enrich_chunk_metadata(chunks: list[Document], raw_documents: list[Document]) -> list[Document]:
    """Requires raw_documents now, to compute each chunk's true position
    within its source document (not just the chunk's own isolated text)."""
    doc_lookup = {doc.metadata["source"]: doc.page_content for doc in raw_documents}
    section_maps = {source: _build_section_map(text) for source, text in doc_lookup.items()}

    enriched = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        full_text = doc_lookup.get(source, "")
        # Find where this chunk's text actually starts in the original document
        offset = full_text.find(chunk.page_content[:50])  # first 50 chars as a fingerprint
        section_title = (
            _section_title_for_offset(section_maps[source], offset)
            if offset != -1 else "(unspecified)"
        )

        new_metadata = {
            **chunk.metadata,
            "document_type": _DOCUMENT_TYPE_MAP.get(source, "unknown"),
            "section_title": section_title,
            "mentions_segment": _detect_segments(chunk.page_content),
        }
        enriched.append(Document(page_content=chunk.page_content, metadata=new_metadata))
    return enriched

