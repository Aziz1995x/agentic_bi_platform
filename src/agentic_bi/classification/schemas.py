"""Structured output contract for question routing.

This enum deliberately mirrors the routing options from the target
architecture's conditional-routing phase (RAG only / SQL only /
SQL+Python / RAG+SQL / full investigation) — the LLM is being asked
to predict a decision the Planner node will act on later, not to
write free text.
"""

from enum import Enum

from pydantic import BaseModel, Field


class InvestigationRoute(str, Enum):
    RAG_ONLY = "rag_only"
    SQL_ONLY = "sql_only"
    SQL_AND_PYTHON = "sql_and_python"
    RAG_AND_SQL = "rag_and_sql"
    FULL_INVESTIGATION = "full_investigation"


class QuestionClassification(BaseModel):
    route: InvestigationRoute = Field(
        description=(
            "Which investigation path this business question requires. "
            "rag_only: pure definition/policy lookup. sql_only: a factual "
            "query answerable directly from the database. sql_and_python: "
            "needs data plus statistical analysis (trends, correlation, "
            "anomaly detection). rag_and_sql: needs both a business "
            "definition and data to apply it. full_investigation: an "
            "open-ended 'why did X happen' question needing multiple "
            "evidence sources."
        )
    )
    reasoning: str = Field(
        description="One-sentence justification for the chosen route."
    )
    requires_human_approval: bool = Field(
        default=False,
        description=(
            "True only if the question implies an action beyond read-only "
            "analysis (e.g. asks the system to change something)."
        ),
    )


class ClassificationBundle(BaseModel):
    """Combined output of the parallel classification step."""
    classification: QuestionClassification
    restated_question: str
