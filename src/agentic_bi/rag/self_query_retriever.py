# src/agentic_bi/rag/self_query_retriever.py -- 

"""SelfQueryRetriever wiring for the enriched-metadata knowledge base.

STATUS: NOT ACTIVE / NOT INTEGRATED. Kept for reference only.

Blocked by cascading import failures between langchain_classic and the
installed langchain_community version's optional vectorstore integrations
(see docs/decisions/decision_log.md, "Self-Query Retrieval -- not
evaluated"). SelfQueryRetriever.from_llm()'s internal translator-selection
logic eagerly imports ~15-20 optional third-party vectorstore classes
(Databricks, DeepLake, and others) purely to support isinstance() checks,
several of which are deprecated/removed from the installed
langchain_community. A compatibility shim was attempted and abandoned
after it introduced its own distinct import-order failure rather than
cleanly resolving the underlying issue.

metadata_enrichment.py (document_type, section_title, mentions_segment)
remains valid, tested, and independently useful -- only the
SelfQueryRetriever wiring itself is inactive.
"""


"""SelfQueryRetriever: LLM translates the question into a metadata filter
plus a semantic search over the remainder, using the structured fields
enrich_chunk_metadata() attached to each chunk.
"""

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore

from agentic_bi.llm.client import get_llm

_METADATA_FIELD_INFO = [
    AttributeInfo(
        name="document_type",
        description="The category of document this chunk comes from: "
                     "'policy' (company rules like refunds, pricing), "
                     "'definitions' (KPI/metric definitions), or "
                     "'schema' (database table/column reference)",
        type="string",
    ),
    AttributeInfo(
        name="section_title",
        description="The markdown section heading this chunk falls under",
        type="string",
    ),
    AttributeInfo(
        name="mentions_segment",
        description="Which customer segments (enterprise, retail) this "
                     "chunk explicitly discusses, if any",
        type="list[string]",
    ),
]

_DOCUMENT_CONTENT_DESCRIPTION = (
    "Company policy documents, KPI/metric definitions, and database "
    "schema reference material for a BI analytics platform"
)


def build_self_query_retriever(vectorstore: VectorStore, k: int = 4) -> SelfQueryRetriever:
    llm = get_llm()
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=_DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=_METADATA_FIELD_INFO,
        search_kwargs={"k": k},
        enable_limit=False,
    )
