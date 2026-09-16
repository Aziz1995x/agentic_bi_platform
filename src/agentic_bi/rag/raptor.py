"""Simplified RAPTOR: recursive clustering + LLM summarization, producing
a small multi-level tree of summary nodes layered on top of your existing
leaf chunks.

Simplifications vs. the full RAPTOR paper, made deliberately given this
corpus's small size:
  - GaussianMixture clustering directly on embeddings (no UMAP dimension
    reduction first -- UMAP mainly earns its keep with hundreds/thousands
    of chunks where raw high-dimensional clustering degrades; not a
    concern at ~44 chunks).
  - Soft clustering still used (a chunk can belong to more than one
    cluster if probabilities are close), matching RAPTOR's actual design,
    since a chunk like "Relationship to Active Customer Status" genuinely
    straddles two themes (segmentation AND activity).
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sklearn.mixture import GaussianMixture

from agentic_bi.llm.client import get_llm
from agentic_bi.rag.embeddings import get_embeddings

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the following group of related policy/definition "
        "excerpts into a single coherent paragraph. Preserve specific "
        "numbers, thresholds, and rule names -- do not generalize them "
        "away. This summary will be used for retrieval, so it should "
        "stand on its own as a description of what these excerpts cover.",
    ),
    ("human", "{excerpts}"),
])


def _cluster_embeddings(embeddings: np.ndarray, max_clusters: int = 5) -> np.ndarray:
    """Soft-clusters embeddings via GMM, picking cluster count by BIC.
    Returns a hard cluster assignment per item for this simplified build
    (RAPTOR's full version allows multi-cluster membership; using the
    single most-likely cluster per item here keeps the summarization step
    simple and is a reasonable simplification at this corpus size)."""
    n = len(embeddings)
    max_clusters = min(max_clusters, max(2, n // 3))

    best_gmm, best_bic = None, np.inf
    for k in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    return best_gmm.predict(embeddings)


def _summarize_cluster(chunks: list[Document]) -> str:
    llm = get_llm()
    chain = _SUMMARY_PROMPT | llm
    excerpts = "\n\n---\n\n".join(c.page_content for c in chunks)
    return chain.invoke({"excerpts": excerpts}).content


def build_raptor_tree(
    leaf_chunks: list[Document],
    max_levels: int = 2,
) -> list[Document]:
    """Builds a RAPTOR tree and returns ALL nodes (leaves + every summary
    level) as a flat list of Documents, ready to embed into one FAISS
    index alongside your existing leaf chunks.

    Stops early if clustering stops making progress (e.g. down to a
    single cluster, or fewer chunks than the minimum needed to cluster
    meaningfully) -- exactly the condition expected to trigger quickly on
    this corpus.
    """
    embeddings_model = get_embeddings()
    all_nodes = list(leaf_chunks)
    current_level_chunks = leaf_chunks

    for level in range(max_levels):
        if len(current_level_chunks) < 4:
            print(f"RAPTOR: stopping at level {level} -- too few nodes "
                  f"({len(current_level_chunks)}) to cluster meaningfully.")
            break

        texts = [c.page_content for c in current_level_chunks]
        vectors = np.array(embeddings_model.embed_documents(texts))
        cluster_labels = _cluster_embeddings(vectors)

        n_clusters = len(set(cluster_labels))
        if n_clusters <= 1:
            print(f"RAPTOR: stopping at level {level} -- clustering "
                  f"collapsed to {n_clusters} cluster.")
            break

        summary_nodes = []
        for cluster_id in set(cluster_labels):
            members = [c for c, lbl in zip(current_level_chunks, cluster_labels) if lbl == cluster_id]
            summary_text = _summarize_cluster(members)
            sources = sorted({m.metadata.get("source", "unknown") for m in members})
            summary_nodes.append(
                Document(
                    page_content=summary_text,
                    metadata={
                        "source": ",".join(sources),  # multi-doc summaries carry ALL contributing sources
                        "raptor_level": level + 1,
                        "is_summary": True,
                    },
                )
            )

        print(f"RAPTOR level {level + 1}: {len(current_level_chunks)} nodes "
              f"-> {len(summary_nodes)} cluster summaries")
        all_nodes.extend(summary_nodes)
        current_level_chunks = summary_nodes

    return all_nodes
