"""The first LCEL chain in the project: prompt -> LLM -> structured output.

Chain of custody for a request through this file:

    ChatPromptTemplate.invoke(input)   -> PromptValue
    PromptValue                        -> passed to the LLM
    LLM (with_structured_output)       -> QuestionClassification instance

`prompt | llm` is LCEL: the `|` operator composes two Runnables into a
RunnableSequence, where the output of the left side becomes the input
of the right side. Nothing here is a "special agent" yet -- it's plain
functional composition, which is exactly why LangChain calls the
building block a "Runnable": anything with .invoke()/.batch()/.stream()
can sit on either side of that pipe.
"""

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate

from agentic_bi.classification.schemas import QuestionClassification, ClassificationBundle, RoutedQuestion
from agentic_bi.llm.client import get_llm

_SYSTEM_PROMPT = """You are the routing component of a BI analyst agent.
Classify the user's business question into exactly one investigation route.
Be decisive -- ambiguous questions should default to full_investigation
rather than guessing a narrower route."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def build_classifier_chain():
    """Returns a Runnable: dict[str, str] -> QuestionClassification."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(QuestionClassification)
    return _prompt | structured_llm


def classify_question(question: str) -> QuestionClassification:
    """Convenience wrapper for a single synchronous call."""
    chain = build_classifier_chain()
    return chain.invoke({"question": question})


_RESTATE_SYSTEM_PROMPT = """Restate the user's business question in one
plain sentence, as if confirming your understanding back to them. Do not
answer it -- only restate it."""

_restate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _RESTATE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def build_bundle_chain():
    """Runs classification and restatement concurrently, merges the result.

    RunnableParallel takes a dict of named branches. Each branch receives
    the SAME input (here: {"question": ...}) and all branches execute
    concurrently -- this is fan-out. The dict returned by .invoke() has
    one key per branch -- this is fan-in. No branch waits on another,
    which matters once one branch is a slow network call (an LLM) and
    another is cheap local logic (RunnableLambda below).
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(QuestionClassification)
    restate_llm = _restate_prompt | llm | RunnableLambda(
        lambda msg: msg.content)

    parallel = RunnableParallel(
        classification=_prompt | structured_llm,
        restated_question=restate_llm,
    )

    def _to_bundle(result: dict) -> ClassificationBundle:
        return ClassificationBundle(**result)
    # print(parallel)
    # print("###########################################################################")
    return parallel | RunnableLambda(_to_bundle)


def classify_question_bundle(question: str) -> ClassificationBundle:
    chain = build_bundle_chain()
    return chain.invoke({"question": question})


def build_routed_chain():
    """Attaches the original input question to the classification output.

    RunnablePassthrough() means "take the input to this step and pass it
    through unchanged." Combined with RunnableParallel, one branch runs
    the real chain (_prompt | structured_llm) and the other branch just
    echoes the input dict back -- no LLM call, no transformation. This is
    the standard LCEL pattern for "I need the model's output AND the
    original input available together downstream," which is exactly what
    a LangGraph state update needs: you rarely want to throw away the
    question just because you've now classified it.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(QuestionClassification)

    parallel = RunnableParallel(
        classification=_prompt | structured_llm,
        original_question=RunnablePassthrough() | RunnableLambda(lambda d: d["question"]),
    )

    def _to_routed(result: dict) -> RoutedQuestion:
        return RoutedQuestion(
            original_question=result["original_question"],
            classification=result["classification"],
        )

    return parallel | RunnableLambda(_to_routed)


def classify_questions_batch(questions: list[str]) -> list[RoutedQuestion]:
    """Runs multiple questions through the routed chain concurrently.

    .batch() is not a for-loop with .invoke() called N times -- it
    dispatches all N inputs concurrently (thread pool under the hood for
    sync chains) and returns results in the same order as the inputs.
    This is the mechanism you'll use in Phase 20 to run an entire
    evaluation dataset (30-50 questions) without writing your own
    concurrency code.
    """
    chain = build_routed_chain()
    return chain.batch([{"question": q} for q in questions])
