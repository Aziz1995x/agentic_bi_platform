"""Prompts for the question-routing chain.

Centralized so Phase 20 (evaluation) and Phase 21 (regression testing)
can diff prompt versions independently of the chain-building code that
consumes them.
"""

from langchain_core.prompts import ChatPromptTemplate

ROUTE_CLASSIFICATION_SYSTEM_PROMPT = """You are the routing component of a BI analyst agent.
Classify the user's business question into exactly one investigation route.
Be decisive -- ambiguous questions should default to full_investigation
rather than guessing a narrower route."""

RESTATE_SYSTEM_PROMPT = """Restate the user's business question in one
plain sentence, as if confirming your understanding back to them. Do not
answer it -- only restate it."""

route_classification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTE_CLASSIFICATION_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

restate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESTATE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)
