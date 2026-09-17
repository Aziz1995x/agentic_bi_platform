"""
captioner.py
────────────
Converts image files to rich text captions via a vision-language model.

Why caption-then-embed rather than direct joint embedding?
──────────────────────────────────────────────────────────
Direct joint embedding (CLIP-style) requires a separate multimodal
embedding model that maps images and text into one shared vector space.
That adds a new model dependency, a different embedding dimension than
text-embedding-3-small, and forces the entire FAISS index to use the
joint model — you can't mix CLIP embeddings with OpenAI text embeddings
in one index without dimension mismatches.

Caption-then-embed sidesteps all of that: the vision model runs ONCE at
ingest time, produces a text description, and everything downstream
(embedding, chunking, FAISS) is identical to what Phase 4 already built.
The cost is that visual detail the captioning model misses is genuinely
lost — but for BI chart retrieval, a GPT-4o-mini caption of a revenue
trend chart captures everything a business query needs.

Caption format
──────────────
We prompt the model for a structured caption that includes:
  1. Chart/table type
  2. What the axes / columns represent
  3. The key business finding
  4. Any anomalies or highlighted elements
  5. The approximate time period

This structure matches what a business user would query — "show me the
return rate by category" maps cleanly onto section 2 and 3 of the
caption.
"""

import base64
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from agentic_bi.llm.multimodal_client import get_multimodal_llm

logger = logging.getLogger(__name__)

# ── Caption prompt ────────────────────────────────────────────────────────────
# Deterministic temperature=0 — captions are facts, not creative writing.
_CAPTION_PROMPT = """\
You are a business intelligence analyst describing a chart or table image \
for a searchable knowledge base.

Describe this image in 150–250 words covering ALL of the following:
1. Visual type (e.g. bar chart, line chart, pie chart, data table, dual-axis chart)
2. What each axis or column represents, including units
3. The primary business finding or trend the visual communicates
4. Any anomalies, highlights, or colour-coded thresholds visible
5. The time period or dataset scope shown

Write in plain prose, not bullet points. Be specific about numbers where \
clearly readable. Do not invent numbers that are not visible in the image. \
Start directly with the description — no preamble.\
"""


def image_to_base64(image_path: Path) -> tuple[str, str]:
    """
    Read an image file and return (base64_data, media_type).

    media_type follows the OpenAI vision API convention:
      PNG  → image/png
      JPG  → image/jpeg
      WEBP → image/webp
    """
    suffix = image_path.suffix.lower()
    media_type_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix)
    if media_type is None:
        raise ValueError(
            f"Unsupported image format: {suffix!r}. "
            f"Supported: {list(media_type_map)}"
        )

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def caption_image(image_path: Path, extra_context: str | None = None) -> str:
    """
    Call the vision LLM to generate a structured caption for one image.

    Parameters
    ----------
    image_path:
        Path to a PNG or JPG file.
    extra_context:
        Optional string appended to the prompt — used to pass in the
        pre-written key_insight from the metadata JSON files.  This helps
        the model anchor its description to the known ground truth rather
        than hallucinating numbers that are hard to read from a small
        thumbnail.

    Returns
    -------
    str
        A 150–250 word plain-prose caption suitable for embedding.
    """
    llm = get_multimodal_llm(temperature=0.0)
    b64_data, media_type = image_to_base64(image_path)

    prompt_text = _CAPTION_PROMPT
    if extra_context:
        prompt_text += (
            f"\n\nAdditional context from the data source "
            f"(use to verify numbers you can see):\n{extra_context}"
        )

    message = HumanMessage(
        content=[
            {
                "type":       "image_url",
                "image_url":  {
                    "url":    f"data:{media_type};base64,{b64_data}",
                    "detail": "high",   # high = full resolution analysis
                },
            },
            {
                "type": "text",
                "text": prompt_text,
            },
        ]
    )

    logger.debug("Captioning %s ...", image_path.name)
    response = llm.invoke([message])
    caption  = response.content.strip()
    logger.debug("Caption length: %d chars", len(caption))
    return caption


def caption_image_safe(
    image_path: Path,
    extra_context: str | None = None,
    fallback: str | None = None,
) -> str:
    """
    caption_image() with error handling.

    If the vision API call fails (network error, rate limit, unsupported
    format), returns `fallback` if provided, otherwise re-raises.
    Used during batch ingestion so one bad image doesn't abort the whole run.
    """
    try:
        return caption_image(image_path, extra_context=extra_context)
    except Exception as exc:
        logger.warning(
            "Failed to caption %s: %s — %s",
            image_path.name,
            type(exc).__name__,
            exc,
        )
        if fallback is not None:
            return fallback
        raise
